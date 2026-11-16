"""Synchronized component timing for the Qwen3-VL dense-latent path.

The timer deliberately lives outside the branch-specific benchmark scripts so
the base, read, grid, and visual variants report the same stage definitions.
Each stage is surrounded by a CUDA synchronize; otherwise CUDA launches would
be measured asynchronously and the component numbers would be meaningless.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Tuple

import torch


def _measure(fn: Callable[[], object], device: torch.device, synchronize: Callable[[torch.device], None]):
    synchronize(device)
    start = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, 1000.0 * (time.perf_counter() - start)


@torch.no_grad()
def timed_model_components(model, prepared, device: torch.device, synchronize, decode_path: str = "sam"):
    """Run one prediction and return logits, latent, and a stage breakdown.

    Definitions used in the paper table:

    * ``qwen_prefill``: one Qwen3-VL forward with ``output_hidden_states``;
      this includes the cost of all native image tokens and internal readout
      tokens in the same forward.
    * ``readout_fusion``: hidden-state layer fusion, token extraction, visual
      pairing, text-spatial attention, and spatial carrier construction.
    * ``latent_adapter``: the 2-D spatial blocks, upsampling, DetailFusion,
      and latent projection.
    * ``sam_image_encoder``: the frozen SAM ViT-H image encoder.
    * ``mask_decoder``: latent-to-prompt projection/calibration and the SAM
      mask decoder.  For ``decode_path=head`` it is the image-free mask head.
    * ``postprocess``: resizing SAM logits back to the original image size.
    """
    _, batch, qwen_inputs, sam_input_sizes, original_sizes = prepared
    decode_path = str(decode_path).lower()
    synchronize(device)
    total_start = time.perf_counter()

    outputs, qwen_ms = _measure(
        lambda: model.qwen(
            **qwen_inputs, output_hidden_states=True, return_dict=True, use_cache=False,
        ),
        device, synchronize,
    )

    def readout_stage():
        hidden = model._fused_hidden(outputs)
        adapter_dtype = model.latent_adapter.output.weight.dtype
        mode = str(getattr(model, "readout_mode", "internal")).lower()
        if mode == "external_query":
            padding_mask = ~qwen_inputs["attention_mask"].bool()
            fused = model.external_spatial_queries(hidden, padding_mask)
            text_states, text_padding = model._read_text_states(
                qwen_inputs["input_ids"], qwen_inputs["attention_mask"], hidden,
                exclude_grid_tokens=False,
            )
            fused = model.text_spatial_attention(
                fused, text_states.to(dtype=adapter_dtype), text_padding,
            )
            vision_map = model._read_external_visual_map(
                qwen_inputs["input_ids"], qwen_inputs["image_grid_thw"], hidden,
            )
            return fused, vision_map, adapter_dtype, mode

        grid_states, vision_states, vision_map = model._read_states(
            qwen_inputs["input_ids"], qwen_inputs["image_grid_thw"], hidden,
        )
        if mode == "visual_only":
            fused = model.latent_adapter.fuse_visual(vision_states.to(dtype=adapter_dtype))
        elif mode == "internal":
            fused = model.latent_adapter.fuse(
                grid_states.to(dtype=adapter_dtype), vision_states.to(dtype=adapter_dtype),
            )
        else:
            raise RuntimeError(
                "Component timing currently supports internal/visual_only readout. "
                f"Received readout_mode={mode!r}."
            )
        text_states, text_padding = model._read_text_states(
            qwen_inputs["input_ids"], qwen_inputs["attention_mask"], hidden,
        )
        fused = model.text_spatial_attention(
            fused, text_states.to(dtype=adapter_dtype), text_padding,
        )
        return fused, vision_map, adapter_dtype, mode

    (fused, vision_map, adapter_dtype, mode), readout_ms = _measure(readout_stage, device, synchronize)

    def latent_stage():
        if mode == "external_query":
            return model.latent_adapter.forward_queries(
                fused, vision_map.to(dtype=adapter_dtype),
            )
        return model.latent_adapter(fused, vision_map.to(dtype=adapter_dtype))

    latent, adapter_ms = _measure(latent_stage, device, synchronize)

    if decode_path == "sam":
        sam_features, sam_ms = _measure(
            lambda: model.teacher.encode_sam_images(batch["sam_images"]), device, synchronize,
        )

        def mask_stage():
            prompt_base = model.teacher.prompt_delta(latent)
            prompt_delta = prompt_base
            if getattr(model, "training_phase", "task_adaptation") == "task_adaptation":
                prompt_delta = prompt_base + model.image_prompt_calibrator(prompt_base, sam_features)
            return model.teacher.decode_prompt(prompt_delta, sam_features)

        masks256, mask_ms = _measure(mask_stage, device, synchronize)
    elif decode_path == "head":
        sam_features, sam_ms = None, 0.0
        masks256, mask_ms = _measure(lambda: model.teacher.decode_mask_head(latent), device, synchronize)
    else:
        raise ValueError(f"decode_path must be 'sam' or 'head', got {decode_path!r}.")

    logits, post_ms = _measure(
        lambda: model.original_logits_from_sizes(masks256, sam_input_sizes, original_sizes),
        device, synchronize,
    )
    synchronize(device)
    total_ms = 1000.0 * (time.perf_counter() - total_start)
    components = {
        "qwen_prefill": qwen_ms,
        "readout_fusion": readout_ms,
        "latent_adapter": adapter_ms,
        "sam_image_encoder": sam_ms,
        "mask_decoder": mask_ms,
        "postprocess": post_ms,
    }
    components["component_sum"] = float(sum(components.values()))
    components["unaccounted"] = float(total_ms - components["component_sum"])
    components["total_model"] = float(total_ms)
    return logits, latent, components




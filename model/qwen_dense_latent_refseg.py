"""llava6-3-8 Stage-2 Student: text-conditioned Qwen readout -> latent -> SAM.

Differences from llava6-3-1, all of them consequences of the failure analysis:

* The 256 readout tokens are fused with the Qwen3-VL *visual* tokens they
  spatially correspond to.  A 512x512 canvas with patch 16 and spatial merge 2
  produces exactly 16x16 visual tokens, so readout token ``i`` and visual token
  ``i`` describe the same 32x32 pixel cell.  llava6-3-1 threw that away.
* The readout is a learnable mixture of several decoder layers instead of the
  last hidden state only.
* The latent adapter upsamples 16x16 -> 32x32 and emits a latent that is
  already projected onto the Teacher manifold, so alignment is a single MSE.
* An explicit no-target head replaces the ``empty_peak`` penalty and the
  "prediction must be exactly all black" inference rule.
* Phase A supervises geometry through the Teacher's image-free mask head, so
  it optimizes IoU-relevant structure instead of a pure latent proxy.
* A text-to-spatial cross-attention residual conditions every grid cell on the
  expression before the latent adapter.
* Phase A uses only latent + image-free head BCE/Dice.  Phase B uses only
  latent + final-mask BCE/Dice + smooth same-category ranking.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from PLAS.model.mask_latent_teacher import (
    LATENT_CHANNELS,
    LATENT_GRID,
    READOUT_GRID,
    MaskLatentTeacher,
    mask_bce_dice,
    normalize_latent,
    relative_latent_error,
    target_for_logits,
)


QWEN_LATENT_GRID_TOKENS = tuple(f"<QWEN_MASK_GRID_{index:03d}>" for index in range(READOUT_GRID ** 2))


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 1):
        if channels % groups == 0:
            return groups
    return 1


def _latent_distance(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample relative latent distance, in the same units as eval error."""
    difference = (predicted.float() - target.float()).square().sum(dim=-3).sqrt()
    reference = target.float().square().sum(dim=-3).sqrt().clamp_min(eps)
    return (difference / reference).flatten(1).mean(dim=1)


def quantize_latent(latent: torch.Tensor, levels: int) -> torch.Tensor:
    """Uniformly quantize every latent element to ``levels`` codebook values.

    ``levels <= 0`` is a no-op.  The bin edges are taken per tensor from the
    observed min/max, so the quantizer never clips the latent it is measuring;
    with ``levels=256`` the error is below bf16 resolution and the output is
    effectively untouched.

    The latent lives on a per-position sphere (``normalize_latent``), and naive
    element-wise rounding moves points off it -- at 4 levels the radius drifts
    from 5.66 to about 8.8, which would corrupt the decoder for reasons that
    have nothing to do with quantization.  Re-normalizing after rounding keeps
    every probe point on the training manifold, so the sweep measures how
    finely the *direction* must be resolved rather than measuring off-manifold
    artifacts.

    This is an inference-time probe for whether the decoder consumes a graded
    value or effectively a small set of codes.  If accuracy is flat down to a
    low level count, the latent carries geometry continuously; a cliff marks
    where the effective resolution of the representation actually sits.
    """
    levels = int(levels)
    if levels <= 0:
        return latent
    if levels == 1:
        return torch.zeros_like(latent)
    low, high = latent.amin(), latent.amax()
    if not torch.isfinite(low) or not torch.isfinite(high) or high <= low:
        return latent
    # Round to the nearest of `levels` evenly spaced points on [low, high].
    step = (high - low) / (levels - 1)
    quantized = (latent - low).div(step).round().clamp_(0, levels - 1).mul_(step).add_(low)
    # Restore the per-position unit direction the decoder was trained on.
    return normalize_latent(quantized)


class SpatialResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.body(value))


class GridTokenEmbedding(nn.Module):
    """Freeze Qwen's vocabulary while exposing exactly 256 trainable rows."""

    def __init__(self, base: nn.Embedding, token_ids: Sequence[int]):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        ids = tuple(map(int, token_ids))
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        lookup = torch.full((base.num_embeddings,), -1, dtype=torch.long)
        lookup[self.token_ids] = torch.arange(len(ids), dtype=torch.long)
        self.register_buffer("lookup", lookup, persistent=False)
        self.rows = nn.Parameter(base.weight.detach()[self.token_ids].clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.base(input_ids)
        rows = self.lookup.to(device=input_ids.device)[input_ids]
        selected = rows.ge(0)
        if bool(selected.any()):
            output = output.clone()
            output[selected] = self.rows[rows[selected]].to(dtype=output.dtype)
        return output


class GridVisionFusion(nn.Module):
    """Fuse the expression-conditioned readout with its co-located visual token.

    The readout stays the base signal: it is the only path that saw the
    referring expression.  The visual token can only *gate in* local image
    evidence for the cell it already corresponds to.
    """

    def __init__(self, hidden_size: int, width: int):
        super().__init__()
        self.grid_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width, bias=False))
        self.vision_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width, bias=False))
        self.gate = nn.Sequential(nn.Linear(2 * width, width), nn.Sigmoid())
        self.norm = nn.LayerNorm(width)

    def forward(self, grid_states: torch.Tensor, vision_states: torch.Tensor) -> torch.Tensor:
        grid = self.grid_projection(grid_states)
        vision = self.vision_projection(vision_states)
        return self.combine_projected(grid, vision)

    def combine_projected(self, grid: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        """Combine already projected branches; used by readout swapping."""
        return self.norm(grid + self.gate(torch.cat((grid, vision), dim=-1)) * vision)


class DetailFusion(nn.Module):
    """Inject native-resolution visual evidence at the *latent* grid.

    The readout grid is 16x16, but the latent is 32x32: one latent cell is 20x20
    original pixels while one Qwen visual token at 512 covers 40x40.  The adapter
    was therefore being asked to invent 4x the spatial detail it received, and
    ``_read_states`` made that worse by average-pooling any finer visual grid
    back down to 16x16.  This module keeps the visual tokens at their native
    resolution and gates them in at 32x32, so raising ``--qwen_image_size``
    actually buys detail instead of being averaged away.

    ``output`` is zero-initialised: at construction the module is the identity,
    so a checkpoint trained without it loads and behaves identically.
    """

    def __init__(self, hidden_size: int, width: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_size, width, 1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Conv2d(2 * width, width, 1), nn.Sigmoid())
        self.body = SpatialResidual(width)
        self.output = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, vision_map: torch.Tensor) -> torch.Tensor:
        if vision_map.shape[-2:] != features.shape[-2:]:
            vision_map = F.interpolate(
                vision_map.float(), size=features.shape[-2:], mode="bilinear", align_corners=False,
            ).to(dtype=features.dtype)
        vision = self.projection(vision_map.to(dtype=features.dtype))
        fused = features + self.gate(torch.cat((features, vision), dim=1)) * vision
        return features + self.output(self.body(fused))


class QwenGridBlock(nn.Module):
    """Bidirectional 2-D mixing after the causal Qwen token readout."""

    def __init__(self, width: int, heads: int, grid: int):
        super().__init__()
        self.grid = int(grid)
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.spatial = SpatialResidual(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        batch, count, channels = tokens.shape
        if count != self.grid ** 2:
            raise RuntimeError(f"Grid block expected {self.grid ** 2} tokens, got {count}.")
        spatial = tokens.transpose(1, 2).reshape(batch, channels, self.grid, self.grid)
        tokens = tokens + self.spatial(spatial).flatten(2).transpose(1, 2)
        return tokens + self.mlp(self.norm2(tokens))


class QwenGridLatentAdapter(nn.Module):
    """Fused 16x16 states -> normalized [B, 32, 32, 32] Teacher latent."""

    def __init__(
        self,
        hidden_size: int,
        width: int = 512,
        depth: int = 4,
        heads: int = 8,
        latent_channels: int = LATENT_CHANNELS,
    ):
        super().__init__()
        if width % heads:
            raise ValueError(f"latent adapter width {width} must be divisible by heads {heads}")
        self.fusion = GridVisionFusion(hidden_size, width)
        self.position = nn.Parameter(torch.zeros(1, READOUT_GRID ** 2, width))
        self.blocks = nn.ModuleList(QwenGridBlock(width, heads, READOUT_GRID) for _ in range(depth))
        upsampled = max(64, width // 2)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, upsampled, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(upsampled), upsampled),
            nn.GELU(),
        )
        self.refine = nn.Sequential(SpatialResidual(upsampled), SpatialResidual(upsampled))
        self.detail = DetailFusion(hidden_size, upsampled)
        self.output = nn.Conv2d(upsampled, int(latent_channels), 1)
        nn.init.trunc_normal_(self.position, std=0.02)

    def fuse(self, grid_states: torch.Tensor, vision_states: torch.Tensor) -> torch.Tensor:
        if grid_states.shape[1] != READOUT_GRID ** 2 or vision_states.shape[1] != READOUT_GRID ** 2:
            raise RuntimeError(
                f"Expected {READOUT_GRID ** 2} readout and visual states, got "
                f"{grid_states.shape[1]} and {vision_states.shape[1]}."
            )
        return self.fusion(grid_states, vision_states) + self.position

    def fuse_projected(self, readout: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        if readout.shape[1:] != (READOUT_GRID ** 2, self.position.shape[-1]):
            raise RuntimeError(f"Readout swap has shape {tuple(readout.shape)}, expected [B,256,{self.position.shape[-1]}].")
        if vision.shape != readout.shape:
            raise RuntimeError(f"Readout swap visual branch shape {tuple(vision.shape)} does not match {tuple(readout.shape)}.")
        return self.fusion.combine_projected(readout, vision) + self.position

    def forward(self, fused: torch.Tensor, vision_map: torch.Tensor = None) -> torch.Tensor:
        tokens = fused
        for block in self.blocks:
            tokens = block(tokens)
        batch, _, channels = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(batch, channels, READOUT_GRID, READOUT_GRID)
        spatial = F.interpolate(spatial.float(), size=(LATENT_GRID, LATENT_GRID), mode="bilinear", align_corners=False)
        spatial = self.upsample(spatial.to(dtype=tokens.dtype))
        spatial = self.refine(spatial)
        if vision_map is not None:
            spatial = self.detail(spatial, vision_map)
        # Emit directly onto the Teacher manifold: alignment becomes one MSE.
        return normalize_latent(self.output(spatial))


class TextSpatialCrossAttention(nn.Module):
    """Condition spatial readout cells on all expression/prompt text tokens.

    The residual output is zero-initialized, so enabling this module is an
    identity mapping at step zero.  It can then learn global relations such as
    left/right, ordinal references, and nearest/farthest comparisons without
    perturbing an already useful latent adapter at initialization.
    """

    def __init__(self, text_hidden_size: int, width: int, heads: int = 8):
        super().__init__()
        if width % int(heads):
            raise ValueError(f"cross-attention width {width} must be divisible by heads {heads}")
        self.query_norm = nn.LayerNorm(width)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_hidden_size),
            nn.Linear(text_hidden_size, width, bias=False),
        )
        self.attention = nn.MultiheadAttention(width, int(heads), batch_first=True)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        text = self.text_projection(text_tokens.to(dtype=spatial_tokens.dtype))
        attended, _ = self.attention(
            self.query_norm(spatial_tokens), text, text,
            key_padding_mask=text_padding_mask, need_weights=False,
        )
        return spatial_tokens + self.output(attended)


class SAMImagePromptCalibrator(nn.Module):
    """Zero-start prompt correction guided by frozen SAM image features."""

    def __init__(
        self, channels: int = 256, width: int = 128, depth: int = 2,
        initial_scale: float = 0.05, max_scale: float = 0.50,
    ):
        super().__init__()
        if int(width) <= 0 or int(depth) <= 0:
            raise ValueError("image prompt calibrator width and depth must be positive.")
        self.max_scale = float(max_scale)
        self.prompt_projection = nn.Sequential(
            nn.Conv2d(channels, width, 1, bias=False), nn.GroupNorm(_group_count(width), width), nn.GELU(),
        )
        self.image_projection = nn.Sequential(
            nn.Conv2d(channels, width, 1, bias=False), nn.GroupNorm(_group_count(width), width), nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Conv2d(2 * width, width, 1), nn.Sigmoid())
        self.body = nn.Sequential(*[SpatialResidual(width) for _ in range(int(depth))])
        self.output = nn.Conv2d(width, channels, 1)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, prompt_base: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        if tuple(prompt_base.shape[1:]) != (256, 64, 64):
            raise RuntimeError(f"Image prompt calibrator expects prompt [B,256,64,64], got {tuple(prompt_base.shape)}")
        if tuple(sam_features.shape) != tuple(prompt_base.shape):
            raise RuntimeError(
                "SAM image feature/prompt mismatch: "
                f"features={tuple(sam_features.shape)}, prompt={tuple(prompt_base.shape)}"
            )
        # Prompt Adapter is intentionally kept in fp32 so its scalar gate is
        # not quantized.  The calibrator may remain bf16, therefore both
        # inputs must be converted to the calibrator's parameter dtype before
        # entering its convolutions.
        calibrator_dtype = next(self.prompt_projection.parameters()).dtype
        prompt_input = prompt_base.to(device=prompt_base.device, dtype=calibrator_dtype)
        image_input = sam_features.to(device=prompt_base.device, dtype=calibrator_dtype)
        prompt = self.prompt_projection(prompt_input)
        image = self.image_projection(image_input)
        fused = prompt + self.gate(torch.cat((prompt, image), dim=1)) * image
        correction = self.output(self.body(fused))
        return self.scale.clamp(0.0, self.max_scale) * torch.tanh(correction)


class QwenLatentStudent(nn.Module):
    """Qwen3-VL -> fused grid readout -> normalized latent -> SAM mask."""

    FORMAT = "llava6_3_8_qwen_latent_student_v1"
    # Older checkpoints can initialize 3-8; the zero-start cross-attention
    # tensors are intentionally absent and remain an identity mapping.
    COMPATIBLE_FORMATS = (
        FORMAT,
        "llava6_3_4_qwen_latent_student_v1",
        "llava6_3_3_qwen_latent_student_v1",
        "llava6_3_2_qwen_latent_student_v1",
    )
    # Modules this branch deleted.  Their tensors in an older checkpoint are
    # dropped on load rather than raising, so a llava6-3-3 Phase-A checkpoint
    # stays usable as a starting point.
    RETIRED_PREFIXES = ("prompt_calibrator.", "no_target_head.")

    def __init__(
        self,
        qwen: nn.Module,
        grid_token_ids: Sequence[int],
        teacher: MaskLatentTeacher,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_mlp: bool = True,
        latent_width: int = 512,
        latent_depth: int = 4,
        text_cross_attention_heads: int = 8,
        readout_layers: Sequence[int] = (-1, -9, -17),
        latent_weight: float = 1.0,
        relative_latent_weight: float = 0.1,
        head_bce_weight: float = 1.0,
        head_dice_weight: float = 1.0,
        mask_bce_weight: float = 2.0,
        mask_dice_weight: float = 2.0,
        smooth_rank_weight: float = 0.5,
        smooth_rank_margin: float = 0.20,
        smooth_rank_temperature: float = 0.05,
        empty_loss_weight: float = 0.5,
        mask_threshold: float = 0.55,
        prompt_calibrator_max_scale: float = 0.50,
        image_prompt_calibrator_width: int = 128,
        image_prompt_calibrator_depth: int = 2,
        image_prompt_calibrator_initial_scale: float = 0.05,
        phase_a_sam_mask_weight: float = 0.0,
        phase_a_sam_bce_weight: float = 0.5,
        phase_a_sam_dice_weight: float = 0.5,
        decoder_only: bool = False,
        freeze_sam_decoder: bool = False,
        sam_decoder_tuning: str = "student_b",
        fp32_prompt_adapter: bool = False,
        training_phase: str = "task_adaptation",
    ):
        super().__init__()
        if len(grid_token_ids) != READOUT_GRID ** 2 or len(set(map(int, grid_token_ids))) != len(grid_token_ids):
            raise ValueError(f"Student requires {READOUT_GRID ** 2} unique grid-token ids.")
        if not 0.0 < float(mask_threshold) < 1.0:
            raise ValueError("mask_threshold must be in (0, 1).")
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.qwen = qwen
        self.qwen.requires_grad_(False)
        self.grid_token_ids = tuple(map(int, grid_token_ids))
        self._enable_qwen_lora(lora_r, lora_alpha, lora_dropout, lora_target_mlp)
        self._install_grid_embedding()

        config = self.qwen.config
        text_config = getattr(config, "text_config", config)
        hidden_size = int(getattr(config, "hidden_size", 0) or getattr(text_config, "hidden_size", 0))
        if hidden_size <= 0:
            raise RuntimeError("Cannot determine the Qwen3-VL text hidden size from the model config.")
        self.hidden_size = hidden_size
        self.text_layers = int(getattr(text_config, "num_hidden_layers", 0))
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(text_config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("Qwen3-VL config does not expose image_token_id; visual fusion needs it.")
        self.image_token_id = int(image_token_id)
        vision_config = getattr(config, "vision_config", None)
        self.spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2) or 2)

        self.readout_layers = tuple(int(layer) for layer in readout_layers)
        if not self.readout_layers:
            raise ValueError("readout_layers must not be empty.")
        self.readout_weights = nn.Parameter(torch.zeros(len(self.readout_layers)))

        self.latent_adapter = QwenGridLatentAdapter(hidden_size, width=latent_width, depth=latent_depth)
        self.text_spatial_attention = TextSpatialCrossAttention(
            hidden_size, latent_width, heads=text_cross_attention_heads,
        )
        # llava6-3-3 trained the image-free calibrator's gate *down* (0.050 ->
        # 0.038) while the image-guided one tripled (0.050 -> 0.181): the model
        # voted the image-free correction useless, so it is gone.
        self.image_prompt_calibrator = SAMImagePromptCalibrator(
            width=image_prompt_calibrator_width,
            depth=image_prompt_calibrator_depth,
            initial_scale=image_prompt_calibrator_initial_scale,
            max_scale=prompt_calibrator_max_scale,
        )

        self.latent_weight = float(latent_weight)
        self.relative_latent_weight = float(relative_latent_weight)
        if self.relative_latent_weight < 0.0:
            raise ValueError("relative_latent_weight must be non-negative.")
        self.head_bce_weight = float(head_bce_weight)
        self.head_dice_weight = float(head_dice_weight)
        self.mask_bce_weight = float(mask_bce_weight)
        self.mask_dice_weight = float(mask_dice_weight)
        self.smooth_rank_weight = float(smooth_rank_weight)
        self.smooth_rank_margin = float(smooth_rank_margin)
        self.smooth_rank_temperature = float(smooth_rank_temperature)
        if self.smooth_rank_temperature <= 0.0:
            raise ValueError("smooth_rank_temperature must be positive.")
        self.empty_loss_weight = float(empty_loss_weight)
        self.mask_threshold = float(mask_threshold)
        self.phase_a_sam_mask_weight = float(phase_a_sam_mask_weight)
        self.phase_a_sam_bce_weight = float(phase_a_sam_bce_weight)
        self.phase_a_sam_dice_weight = float(phase_a_sam_dice_weight)
        self.decoder_only = bool(decoder_only)
        self.freeze_sam_decoder = bool(freeze_sam_decoder)
        self.sam_decoder_tuning = str(sam_decoder_tuning)
        if self.sam_decoder_tuning not in {"student_b", "student_b_full"}:
            raise ValueError("sam_decoder_tuning must be 'student_b' or 'student_b_full'.")
        self.fp32_prompt_adapter = bool(fp32_prompt_adapter)
        self.set_training_phase(training_phase)

    # ------------------------------------------------------------- lifecycle

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"latent_warmup", "task_adaptation"}:
            raise ValueError("training_phase must be 'latent_warmup' or 'task_adaptation'.")
        self.training_phase = phase
        task_adaptation = phase == "task_adaptation"
        # RefCOCO/RefCOCO+/RefCOCOg train splits contain positive referring
        # expressions only.  With a zero loss weight, keep this compatibility
        # head frozen rather than silently optimizing it toward a meaningless
        # all-present prior.  gRefCOCO runs retain the explicit classifier.
        # Phase A never touches SAM, so the Teacher stays completely frozen and
        # its mask head provides image-free geometry supervision.
        tuning_mode = self.sam_decoder_tuning if task_adaptation else "freeze"
        self.teacher.configure_tuning(tuning_mode)
        self.image_prompt_calibrator.requires_grad_(task_adaptation)
        if task_adaptation and self.decoder_only:
            # Controlled Phase-B ablation: keep the Phase-A latent fixed and
            # adapt only the prompt/SAM side to the student's latent manifold.
            self.qwen.requires_grad_(False)
            self._grid_embedding.requires_grad_(False)
            self.latent_adapter.requires_grad_(False)
            self.text_spatial_attention.requires_grad_(False)
            self.readout_weights.requires_grad_(False)
        if task_adaptation and self.freeze_sam_decoder:
            # Prompt-Adapter-focused Phase-B: keep the pretrained SAM decoder
            # fixed while adapting the latent-to-dense-prompt mapping.
            self.teacher.sam.mask_decoder.requires_grad_(False)

    def _enable_qwen_lora(self, rank: int, alpha: int, dropout: float, target_mlp: bool) -> None:
        if int(rank) <= 0:
            raise ValueError("qwen_lora_r must be positive.")
        suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if bool(target_mlp):
            suffixes += ["gate_proj", "up_proj", "down_proj"]
        targets = [
            name for name, module in self.qwen.named_modules()
            if isinstance(module, nn.Linear) and "visual" not in name and name.split(".")[-1] in suffixes
        ]
        if not targets:
            raise RuntimeError(
                "No non-visual Qwen projection modules found. Check the installed Qwen3-VL Transformers version."
            )
        self.qwen = get_peft_model(
            self.qwen,
            LoraConfig(
                r=int(rank), lora_alpha=int(alpha), lora_dropout=float(dropout),
                bias="none", task_type="CAUSAL_LM", target_modules=targets,
            ),
        )
        try:
            self.qwen.enable_input_require_grads()
        except AttributeError:
            pass
        self.qwen.config.use_cache = False

    def _install_grid_embedding(self) -> None:
        base = self.qwen.get_input_embeddings()
        if not isinstance(base, nn.Embedding):
            raise RuntimeError(f"Expected a Qwen input embedding nn.Embedding, got {type(base)!r}.")
        replacement = GridTokenEmbedding(base, self.grid_token_ids)

        def replace(module: nn.Module) -> bool:
            for name, child in module.named_children():
                if child is base:
                    setattr(module, name, replacement)
                    return True
                if replace(child):
                    return True
            return False

        if not replace(self.qwen):
            raise RuntimeError("Could not replace the Qwen input embedding with the grid-token wrapper.")
        object.__setattr__(self, "_grid_embedding", replacement)

    def train(self, mode: bool = True):
        super().train(mode)
        self.qwen.train(mode)
        self.teacher.train(mode)
        return self

    @property
    def config(self):
        return self.qwen.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {})
        except TypeError:
            self.qwen.gradient_checkpointing_enable()
        try:
            self.qwen.enable_input_require_grads()
        except AttributeError:
            pass
        self.qwen.config.use_cache = False

    def gradient_checkpointing_disable(self):
        self.qwen.gradient_checkpointing_disable()

    def restore_fp32_modules(self) -> list:
        """Keep the frozen latent target/decoding path in fp32.

        DeepSpeed casts the whole module to bf16 at engine construction.  These
        modules carry no optimizer state, so restoring fp32 afterwards is safe
        and removes bf16 quantization from both sides of the latent regression.
        """
        restored = []
        # The Prompt Adapter is trainable in Phase-B, but its scalar gate must
        # not be left in bf16: at ~0.19 the bf16 quantization step is much
        # larger than the Adam updates and the logged value can appear frozen.
        if self.fp32_prompt_adapter:
            self.teacher.prompt_adapter.float()
            restored.append("teacher.prompt_adapter")
        for name in ("mask_encoder", "mask_head"):
            module = getattr(self.teacher, name)
            if all(not parameter.requires_grad for parameter in module.parameters()):
                module.float()
                restored.append(f"teacher.{name}")
        return restored

    # ---------------------------------------------------------------- readout

    def _fused_hidden(self, outputs) -> torch.Tensor:
        states = outputs.hidden_states
        weights = torch.softmax(self.readout_weights.float(), dim=0)
        fused = None
        for weight, layer in zip(weights, self.readout_layers):
            value = states[layer].float() * weight
            fused = value if fused is None else fused + value
        return fused

    def _read_states(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor, hidden: torch.Tensor):
        """Return (grid_states, vision_states, vision_map).

        The first two are [B, 256, hidden] on the 16x16 readout grid; the third
        is [B, hidden, 32, 32] at the latent grid, carrying whatever resolution
        the Qwen processor actually produced.
        """
        if hidden.shape[:2] != input_ids.shape:
            raise RuntimeError(
                "Qwen3-VL hidden states do not align with input_ids. "
                "Use transformers>=4.57 and the official Qwen3-VL processor."
            )
        expected = torch.tensor(self.grid_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        grid_values, vision_values, detail_values = [], [], []
        for batch in range(input_ids.shape[0]):
            positions = torch.nonzero(torch.isin(input_ids[batch], expected), as_tuple=False).flatten()
            if positions.numel() != len(self.grid_token_ids):
                raise RuntimeError(f"Sample {batch} has {positions.numel()} grid tokens, expected {len(self.grid_token_ids)}.")
            if not torch.equal(input_ids[batch, positions], expected):
                raise RuntimeError("Grid tokens were reordered or corrupted by tokenization.")
            grid_values.append(hidden[batch, positions])

            image_positions = torch.nonzero(input_ids[batch].eq(self.image_token_id), as_tuple=False).flatten()
            rows = int(image_grid_thw[batch][1]) // self.spatial_merge_size
            columns = int(image_grid_thw[batch][2]) // self.spatial_merge_size
            if image_positions.numel() != rows * columns:
                raise RuntimeError(
                    f"Sample {batch} has {image_positions.numel()} image tokens but grid_thw implies {rows * columns}."
                )
            visual = hidden[batch, image_positions].reshape(rows, columns, hidden.shape[-1])
            visual = visual.permute(2, 0, 1).unsqueeze(0)
            # The 16x16 copy pairs one visual token with one readout token.
            pooled = visual if (rows, columns) == (READOUT_GRID, READOUT_GRID) else                 F.adaptive_avg_pool2d(visual.float(), (READOUT_GRID, READOUT_GRID)).to(dtype=visual.dtype)
            vision_values.append(pooled.flatten(2).transpose(1, 2)[0])
            # The native-resolution copy is resampled straight onto the latent
            # grid.  Both live on the same SAM canvas, so a plain bilinear
            # resample is geometrically exact -- no offset correction needed.
            detail = visual if (rows, columns) == (LATENT_GRID, LATENT_GRID) else F.interpolate(
                visual.float(), size=(LATENT_GRID, LATENT_GRID), mode="bilinear", align_corners=False,
            ).to(dtype=visual.dtype)
            detail_values.append(detail[0])
        return torch.stack(grid_values), torch.stack(vision_values), torch.stack(detail_values)

    def _read_text_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, hidden: torch.Tensor,
    ):
        """Pad non-image/non-grid token states for text-to-spatial attention."""
        grid_ids = torch.tensor(self.grid_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        valid = attention_mask.bool() & input_ids.ne(self.image_token_id) & ~torch.isin(input_ids, grid_ids)
        lengths = valid.sum(dim=1)
        if bool(lengths.eq(0).any()):
            raise RuntimeError("Every sample must contain at least one text token.")
        maximum = int(lengths.max())
        values = hidden.new_zeros((hidden.shape[0], maximum, hidden.shape[-1]))
        padding = torch.ones((hidden.shape[0], maximum), device=hidden.device, dtype=torch.bool)
        for batch in range(hidden.shape[0]):
            selected = hidden[batch, valid[batch]]
            values[batch, : selected.shape[0]] = selected
            padding[batch, : selected.shape[0]] = False
        return values, padding

    def encode_tokens(self, qwen_inputs: Dict[str, torch.Tensor], return_probe_features: bool = False,
                      override_readout: torch.Tensor = None):
        """Run Qwen once; optionally expose pre-adapter spatial probe features."""
        outputs = self.qwen(**qwen_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = self._fused_hidden(outputs)
        grid_states, vision_states, vision_map = self._read_states(
            qwen_inputs["input_ids"], qwen_inputs["image_grid_thw"], hidden,
        )
        adapter_dtype = self.latent_adapter.output.weight.dtype
        readout_projected = self.latent_adapter.fusion.grid_projection(grid_states.to(dtype=adapter_dtype))
        vision_projected = self.latent_adapter.fusion.vision_projection(vision_states.to(dtype=adapter_dtype))
        if override_readout is not None:
            override_readout = override_readout.to(device=readout_projected.device, dtype=adapter_dtype)
            if override_readout.shape != readout_projected.shape:
                raise RuntimeError(
                    f"override_readout has shape {tuple(override_readout.shape)}, "
                    f"expected {tuple(readout_projected.shape)}."
                )
            readout_projected = override_readout
        fused = self.latent_adapter.fuse_projected(readout_projected, vision_projected)
        text_states, text_padding = self._read_text_states(
            qwen_inputs["input_ids"], qwen_inputs["attention_mask"], hidden,
        )
        fused = self.text_spatial_attention(
            fused, text_states.to(dtype=adapter_dtype), text_padding,
        )
        latent = self.latent_adapter(fused, vision_map.to(dtype=adapter_dtype))
        if not return_probe_features:
            return latent, fused
        # All three probes have the same 16x16 grid and adapter width.  The
        # readout/visual inputs are separated *before* GridVisionFusion, while
        # joint is exactly the tensor passed into the latent adapter.  Native
        # image-token states precede the referring expression in the causal
        # sequence; this is intentional for the visual-only control.
        probe = {
            "readout": readout_projected,
            "visual": vision_projected,
            "joint": fused,
        }
        expected = torch.tensor(self.grid_token_ids, device=qwen_inputs["input_ids"].device,
                                dtype=qwen_inputs["input_ids"].dtype)
        positions = []
        for sample_ids in qwen_inputs["input_ids"]:
            selected = torch.nonzero(torch.isin(sample_ids, expected), as_tuple=False).flatten()
            if selected.numel() != len(self.grid_token_ids) or not torch.equal(sample_ids[selected], expected):
                raise RuntimeError("Spatial readout token positions do not match the configured token IDs.")
            positions.append(selected)
        probe["readout_layers"] = {
            str(layer): torch.stack([
                outputs.hidden_states[layer][batch, token_positions]
                for batch, token_positions in enumerate(positions)
            ]) for layer in self.readout_layers
        }
        return latent, fused, probe

    def encode(self, qwen_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """RefCOCO/RefCOCO+/RefCOCOg have no no-target expressions, so the
        Student emits a latent and nothing else."""
        return self.encode_tokens(qwen_inputs)[0]

    def predict_latent(self, qwen_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_tokens(qwen_inputs)[0]

    # --------------------------------------------------------------- decoding

    def decode_with_sam_features(self, latent: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        prompt_base = self.teacher.prompt_delta(latent)
        prompt_delta = prompt_base
        if self.training_phase == "task_adaptation":
            prompt_delta = prompt_base + self.image_prompt_calibrator(prompt_base, sam_features)
        return self.teacher.decode_prompt(prompt_delta, sam_features)

    def original_logits_from_sizes(self, masks256, sam_input_sizes, original_sizes):
        return self.teacher.original_logits(masks256, sam_input_sizes, original_sizes)

    def decode_with_head(self, latent: torch.Tensor) -> torch.Tensor:
        """Image-free decoding: latent -> 256x256 mask logits, no SAM features."""
        return self.teacher.decode_mask_head(latent)

    def decode_latent(self, latent: torch.Tensor, sam_images, decode_path: str,
                      latent_levels: int = 0) -> torch.Tensor:
        """Route one latent through the selected decoder.

        ``sam``   -- latent -> dense prompt -> SAM mask decoder (deployment path).
        ``head``  -- latent -> image-free mask head (Stage-I reconstruction path).

        Both branches take the same latent, so an evaluation that toggles this
        argument isolates the decoder from the latent representation.
        ``latent_levels`` additionally quantizes the latent before decoding.
        """
        latent = quantize_latent(latent, latent_levels)
        decode_path = str(decode_path).lower()
        if decode_path == "head":
            return self.decode_with_head(latent)
        if decode_path != "sam":
            raise ValueError(f"decode_path must be 'sam' or 'head', got {decode_path!r}.")
        sam_features = self.teacher.encode_sam_images(sam_images)
        return self.decode_with_sam_features(latent, sam_features)

    @torch.no_grad()
    def predict(self, qwen_inputs, sam_images, sam_input_sizes, original_sizes,
                decode_path: str = "sam", latent_levels: int = 0,
                return_probe_features: bool = False,
                override_readout: torch.Tensor = None):
        """Inference: full-resolution logits and the latent that produced them."""
        encoded = self.encode_tokens(
            qwen_inputs, return_probe_features=return_probe_features,
            override_readout=override_readout,
        )
        latent = encoded[0]
        masks256 = self.decode_latent(latent, sam_images, decode_path, latent_levels)
        logits = self.original_logits_from_sizes(masks256, sam_input_sizes, original_sizes)
        return (logits, latent, encoded[2]) if return_probe_features else (logits, latent)

    # ------------------------------------------------------------------ losses

    @staticmethod
    def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def forward(
        self,
        qwen_inputs,
        sam_images,
        gt_masks,
        original_sizes,
        sam_input_sizes,
        target_groups=None,
        distractor_latents=None,
        distractor_valid=None,
        compute_mask_losses: bool = True,
        **_,
    ) -> Dict[str, torch.Tensor]:
        # Run frozen image encoders before constructing Qwen's backward graph.
        # At 1024, SAM ViT-H needs a large temporary attention tensor; running
        # it after Qwen made both activation peaks overlap and caused 24-GB
        # cards to OOM even with a micro-batch of one.
        phase_a_sam = self.training_phase == "latent_warmup" and self.phase_a_sam_mask_weight > 0.0
        need_sam_features = compute_mask_losses or phase_a_sam
        need_latent_target = (
            self.latent_weight != 0.0
            or self.relative_latent_weight != 0.0
            or not compute_mask_losses
            or (
                self.smooth_rank_weight != 0.0
                and distractor_latents is not None
                and distractor_valid is not None
            )
        )
        with torch.no_grad():
            if need_latent_target:
                target_latent, mask256 = self.teacher.encode_gt(gt_masks, sam_input_sizes)
            else:
                target_latent, mask256 = None, None
            sam_features = self.teacher.encode_sam_images(sam_images) if need_sam_features else None
        latent = self.encode_tokens(qwen_inputs)[0]
        if target_latent is not None:
            target_latent = target_latent.to(device=latent.device)
        if mask256 is not None:
            mask256 = mask256.to(device=latent.device)

        if target_groups is None:
            target_groups = torch.zeros(latent.shape[0], device=latent.device, dtype=torch.long)
        empty = target_groups.to(device=latent.device).eq(2)
        pixel_weights = torch.where(empty, latent.new_full((), self.empty_loss_weight), latent.new_ones(()))

        zero = latent.float().sum() * 0.0
        # 1. Latent alignment. In a strict mask-only run there is no Teacher
        # latent target at all; the zero metrics preserve the log schema while
        # ensuring no latent objective or random target can affect training.
        if target_latent is None:
            latent_loss = zero
            relative_latent_loss = zero
            latent_error = zero.detach()
        else:
            latent_loss = F.mse_loss(latent.float(), target_latent.float())
            # Keep the live relative error in `loss` when enabled so gradients
            # reach Qwen/adapter parameters; log a detached copy as a metric.
            relative_latent_loss = relative_latent_error(latent, target_latent)
            latent_error = relative_latent_loss.detach()

        head_bce, head_dice = zero, zero
        loss = (
            self.latent_weight * latent_loss
            + self.relative_latent_weight * relative_latent_loss
        )
        if not compute_mask_losses:
            # Phase A uses latent MSE + relative latent error plus image-free
            # head BCE/Dice.  No SAM-path or ranking term is evaluated.
            head_logits = self.teacher.decode_mask_head(latent)
            per_bce, per_dice = mask_bce_dice(head_logits, mask256)
            head_bce = self._weighted(per_bce, pixel_weights)
            head_dice = self._weighted(per_dice, pixel_weights)
            loss = loss + self.head_bce_weight * head_bce + self.head_dice_weight * head_dice
        terms = {
            "loss_latent": latent_loss, "loss_latent_error": latent_error,
            "loss_head_bce": head_bce, "loss_head_dice": head_dice,
            "loss_mask_bce": zero, "loss_mask_dice": zero,
            "loss_smooth_rank": zero, "loss_smooth_rank_violation_rate": zero,
        }
        if not compute_mask_losses and not phase_a_sam:
            terms["loss"] = loss
            return terms

        # Phase B uses latent MSE + relative latent error + mask BCE + mask Dice.
        # The optional smooth
        # same-category ranking term is supported by the model, but the 3-8
        # recipe disables it and supplies no distractor cache.  Head and
        # boundary terms are absent.
        if sam_features is None:
            raise RuntimeError("Phase B requires precomputed SAM image features.")
        masks256 = self.decode_with_sam_features(latent, sam_features)
        logits = self.original_logits_from_sizes(masks256, sam_input_sizes, original_sizes)
        sample_bce, sample_dice = [], []
        for logit, mask in zip(logits, gt_masks):
            bce, dice = mask_bce_dice(logit, target_for_logits(logit, mask))
            sample_bce.append(bce.mean())
            sample_dice.append(dice.mean())
        mask_bce = self._weighted(torch.stack(sample_bce), pixel_weights)
        mask_dice = self._weighted(torch.stack(sample_dice), pixel_weights)
        smooth_rank, violation_rate = zero, zero
        if (
            self.smooth_rank_weight != 0.0
            and target_latent is not None
            and distractor_latents is not None
            and distractor_valid is not None
            and bool(distractor_valid.any())
        ):
            distractors = distractor_latents.to(device=latent.device, dtype=torch.float32)
            valid = distractor_valid.to(device=latent.device).bool()
            positive = _latent_distance(latent, target_latent)
            repeated = latent.unsqueeze(1).expand_as(distractors).reshape(-1, *latent.shape[1:])
            negative = _latent_distance(
                repeated, distractors.reshape(-1, *latent.shape[1:]),
            ).view(distractors.shape[:2])
            negative = negative.masked_fill(~valid, float("inf"))
            nearest = negative.min(dim=1).values
            present = valid.any(dim=1)
            score = self.smooth_rank_margin + positive - nearest
            per_sample = self.smooth_rank_temperature * F.softplus(score / self.smooth_rank_temperature)
            smooth_rank = per_sample[present].mean()
            violation_rate = score[present].gt(0).float().mean().detach()

        if phase_a_sam:
            loss = (
                loss
                + self.phase_a_sam_mask_weight * (
                    self.phase_a_sam_bce_weight * mask_bce
                    + self.phase_a_sam_dice_weight * mask_dice
                )
            )
        else:
            loss = (
                self.latent_weight * latent_loss
                + self.relative_latent_weight * relative_latent_loss
                + self.mask_bce_weight * mask_bce
                + self.mask_dice_weight * mask_dice
                + self.smooth_rank_weight * smooth_rank
            )
        terms.update({
            "loss": loss, "loss_mask_bce": mask_bce, "loss_mask_dice": mask_dice,
            "loss_smooth_rank": smooth_rank,
            "loss_smooth_rank_violation_rate": violation_rate,
        })
        return terms

    # --------------------------------------------------------- parameter sets

    def qwen_parameter_names(self):
        grid_parameter = self._grid_embedding.rows
        return {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("qwen.") and parameter is not grid_parameter
        }

    def grid_embedding_parameter_ids(self):
        return {id(self._grid_embedding.rows)}

    def adapter_parameter_names(self):
        return {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and (
                name.startswith("latent_adapter.")
                or name.startswith("text_spatial_attention.")
                or name.startswith("image_prompt_calibrator.")
                or name == "readout_weights"
            )
        }

    def calibrator_scale_parameter_names(self):
        """The scalar that gates the whole prompt correction.

        At the adapter LR they crawl (0.050 -> 0.095 over a full epoch), so they
        get their own, much larger step size.
        """
        return {"image_prompt_calibrator.scale"}

    def prompt_adapter_scale_parameter_names(self):
        return {"teacher.prompt_adapter.delta_scale"}

    def prompt_adapter_parameter_names(self):
        return {f"teacher.{name}" for name in self.teacher.prompt_adapter_parameter_names()}

    def decoder_parameter_names(self):
        return {f"teacher.{name}" for name in self.teacher.sam_decoder_parameter_names()}

    def trainable_summary(self):
        decoder_total = sum(p.numel() for p in self.teacher.sam.mask_decoder.parameters())
        decoder_trainable = sum(p.numel() for p in self.teacher.sam.mask_decoder.parameters() if p.requires_grad)
        return {
            "qwen_lora": sum(
                parameter.numel() for name, parameter in self.named_parameters()
                if parameter.requires_grad and name.startswith("qwen.") and parameter is not self._grid_embedding.rows
            ),
            "grid_token_rows": self._grid_embedding.rows.numel(),
            "latent_adapter": sum(p.numel() for p in self.latent_adapter.parameters() if p.requires_grad),
            "text_spatial_attention": sum(
                p.numel() for p in self.text_spatial_attention.parameters() if p.requires_grad
            ),
            "image_prompt_calibrator": sum(
                p.numel() for p in self.image_prompt_calibrator.parameters() if p.requires_grad
            ),
            "teacher_prompt_adapter": sum(p.numel() for p in self.teacher.prompt_adapter.parameters() if p.requires_grad),
            # Keep the old key for downstream log parsers; the value is no
            # longer necessarily a half-decoder count when student_b_full is
            # selected.
            "sam_decoder_half": decoder_trainable,
            "sam_decoder_trainable": decoder_trainable,
            "sam_decoder_total": decoder_total,
            "sam_decoder_trainable_percent": round(100.0 * decoder_trainable / max(1, decoder_total), 2),
            "sam_decoder_tuning": self.sam_decoder_tuning if self.training_phase == "task_adaptation" else "freeze",
        }

    def calibration_scales(self):
        return {
            "image_prompt_calibrator_scale": float(self.image_prompt_calibrator.scale.detach().float().clamp(0.0, self.image_prompt_calibrator.max_scale)),
            "prompt_adapter_delta_scale": float(self.teacher.prompt_adapter.delta_scale.detach().float().clamp(0.0, 0.5)),
            "prompt_adapter_delta_scale_raw": float(self.teacher.prompt_adapter.delta_scale.detach().float()),
            "readout_weights": [float(value) for value in torch.softmax(self.readout_weights.detach().float(), dim=0)],
        }

    # ------------------------------------------------------------ checkpoints

    def save_student(self, path: str) -> None:
        trainable_names = {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and parameter is not self._grid_embedding.rows
        }
        if self.decoder_only:
            # Decoder-only Phase-B checkpoints must remain self-contained: the
            # frozen Phase-A latent path is not trainable in this run, so it
            # would otherwise be omitted and evaluation would instantiate it
            # randomly.  Save its learned LoRA/adapter state as well.
            trainable_names.update(
                name for name, parameter in self.named_parameters()
                if name.startswith(("qwen.", "latent_adapter.", "text_spatial_attention."))
                or name == "readout_weights"
            )
        trainable = {
            name: value.detach().cpu() for name, value in self.state_dict().items() if name in trainable_names
        }
        torch.save({
            "format": self.FORMAT,
            "state_dict": trainable,
            "grid_token_ids": list(self.grid_token_ids),
            "grid_embeddings": self._grid_embedding.rows.detach().cpu(),
            "readout_layers": list(self.readout_layers),
            "training_phase": self.training_phase,
        }, path)

    def load_student(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") not in self.COMPATIBLE_FORMATS:
            raise RuntimeError(
                f"Not a compatible Student checkpoint (need one of {self.COMPATIBLE_FORMATS}): {path}"
            )
        if checkpoint.get("grid_token_ids") != list(self.grid_token_ids):
            raise RuntimeError("Student grid-token ids do not match this tokenizer.")
        if list(checkpoint.get("readout_layers", self.readout_layers)) != list(self.readout_layers):
            raise RuntimeError(
                f"Checkpoint readout layers {checkpoint.get('readout_layers')} differ from {list(self.readout_layers)}."
            )
        own = self.state_dict()
        retired = [name for name in checkpoint["state_dict"] if name.startswith(self.RETIRED_PREFIXES)]
        unknown = [
            name for name in checkpoint["state_dict"]
            if name not in own and name not in retired
        ]
        if unknown:
            raise RuntimeError(f"Student checkpoint has unexpected tensors: {unknown[:4]}")
        if retired:
            print(f"Dropped {len(retired)} tensors from modules this branch removed (e.g. {retired[0]}).")
        shape_mismatches = [
            (name, tuple(own[name].shape), tuple(value.shape))
            for name, value in checkpoint["state_dict"].items()
            if name not in retired and name in own and tuple(own[name].shape) != tuple(value.shape)
        ]
        if shape_mismatches:
            preview = "; ".join(
                f"{name}: current={current}, checkpoint={saved}"
                for name, current, saved in shape_mismatches[:4]
            )
            width_hint = ""
            for name, current, saved in shape_mismatches:
                if name == "latent_adapter.position" and len(saved) == 3:
                    width_hint = f" Set --latent_width {saved[-1]} to match this checkpoint."
                    break
            raise RuntimeError(
                "Student checkpoint architecture mismatch. "
                f"{preview}.{width_hint}"
            )
        for name, value in checkpoint["state_dict"].items():
            if name in retired:
                continue
            own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))
        rows = checkpoint.get("grid_embeddings")
        expected_shape = tuple(self._grid_embedding.rows.shape)
        if rows is None or tuple(rows.shape) != expected_shape:
            raise RuntimeError(
                f"Grid embedding rows have shape {None if rows is None else tuple(rows.shape)}, expected {expected_shape}."
            )
        with torch.no_grad():
            self._grid_embedding.rows.copy_(
                rows.to(device=self._grid_embedding.rows.device, dtype=self._grid_embedding.rows.dtype)
            )






"""llava6-3-4 Stage-1: a *regressable* mask latent.

The llava6-3-1 Teacher reached ~93 gIoU as a reconstruction oracle but its
latent was an unconstrained 256x16x16 autoencoder code: nothing forced small
latent errors to become small mask errors, so the Stage-2 Student lost ~20
points the moment it had to regress that code.  This module keeps the same
interface contract (GT mask -> latent -> dense SAM prompt -> mask) and changes
only what made the latent hard to predict:

1. The latent is normalized per spatial position (unit direction scaled so the
   per-element RMS is 1).  The regression target is bounded and isotropic, so a
   single MSE replaces the four-term direction/radius/RMS alignment loss.
2. The latent is 32 channels on a 32x32 grid instead of 256 channels on 16x16.
   Half the values, four times the spatial resolution -- spatial resolution is
   what referring segmentation actually needs.
3. The decoding path is trained under latent noise, so it must stay accurate
   inside the error ball the Student actually lands in.
4. A second, *image-independent* decoder (MaskLatentDecoder) maps the latent
   straight back to a 256x256 mask.  Stage-2 Phase A uses it for true BCE/Dice
   geometry supervision without running SAM's ViT-H image encoder.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PLAS.model.segment_anything import sam_model_registry
from PLAS.model.segment_anything.modeling.common import LayerNorm2d


# 32 * 32 * 32 = 32,768 continuous values (half of llava6-3-1) at 4x the
# spatial resolution of the old native-16 latent.
LATENT_GRID = 32
LATENT_CHANNELS = 32
# The Qwen Student still reads 16x16 = 256 tokens, which is exactly the Qwen3-VL
# visual-token grid for a 512x512 canvas.  The Student adapter upsamples 16->32.
READOUT_GRID = 16
MASK_CANVAS = 256
PROMPT_GRADIENT_CLIP = 1.0
PARAMETER_GRADIENT_CLIP = 1.0


def normalize_latent(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project onto the latent manifold: unit direction per spatial position.

    Scaling by sqrt(C) keeps the per-element RMS at 1.0, so a plain MSE on this
    tensor is numerically well scaled *and* is exactly ``2 * (1 - cosine)``.
    """
    channels = value.shape[1]
    return F.normalize(value.float(), dim=1, eps=eps) * math.sqrt(float(channels))


def perturb_latent(latent: torch.Tensor, sigma) -> torch.Tensor:
    """Add relative noise of magnitude ``sigma`` and re-project on the manifold."""
    if isinstance(sigma, (int, float)):
        if float(sigma) <= 0.0:
            return latent
        sigma = latent.new_full((latent.shape[0],), float(sigma))
    sigma = sigma.to(device=latent.device, dtype=latent.dtype).view(-1, 1, 1, 1)
    if bool((sigma <= 0).all()):
        return latent
    # Per position ||noise|| / ||latent|| == sigma, because the latent has
    # per-element scale 1 and the noise has per-element scale sigma.
    return normalize_latent(latent + torch.randn_like(latent) * sigma)


def relative_latent_error(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean per-position ||dz|| / ||z||: the ball Stage-1 noise has to cover."""
    difference = (predicted.float() - target.float()).square().sum(dim=1).sqrt()
    reference = target.float().square().sum(dim=1).sqrt().clamp_min(eps)
    return (difference / reference).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    # The +1 makes empty-GT/empty-prediction a correct reconstruction while
    # still penalizing foreground leakage on no-target expressions.
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def mask_bce_dice(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample BCE and Dice.  Callers decide how to weight the batch."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3))
    return bce, dice_loss(logits, target)


def target_for_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=torch.float32)
    if target.ndim == logits.ndim - 1:
        target = target.unsqueeze(1)
    if tuple(target.shape) != tuple(logits.shape):
        raise RuntimeError(f"Mask target/logit mismatch: target={tuple(target.shape)}, logits={tuple(logits.shape)}")
    return target


def boundary_bce_loss(logits: torch.Tensor, target: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """BCE restricted to a thin GT boundary band; empty masks contribute zero."""
    if int(radius) <= 0:
        raise ValueError("boundary radius must be positive.")
    kernel = 2 * int(radius) + 1
    target = target.float()
    dilated = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=kernel, stride=1, padding=radius)
    band = (dilated - eroded).clamp_(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (bce * band).sum() / band.sum().clamp_min(1.0)


def sanitize_prompt_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        gradient, nan=0.0, posinf=PROMPT_GRADIENT_CLIP, neginf=-PROMPT_GRADIENT_CLIP,
    ).clamp_(-PROMPT_GRADIENT_CLIP, PROMPT_GRADIENT_CLIP)


def sanitize_parameter_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        gradient, nan=0.0, posinf=PARAMETER_GRADIENT_CLIP, neginf=-PARAMETER_GRADIENT_CLIP,
    ).clamp_(-PARAMETER_GRADIENT_CLIP, PARAMETER_GRADIENT_CLIP)


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
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


class MaskLatentEncoder(nn.Module):
    """256x256 mask -> normalized [B, 32, 32, 32] latent."""

    def __init__(self, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, bias=False), LayerNorm2d(32), nn.GELU(), ResidualBlock(32),
        )
        self.down128 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False), LayerNorm2d(64), nn.GELU(), ResidualBlock(64),
        )
        self.down64 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False), LayerNorm2d(128), nn.GELU(), ResidualBlock(128),
        )
        self.project = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), LayerNorm2d(128), nn.GELU(),
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128),
            nn.Conv2d(128, int(latent_channels), 1),
        )

    def forward(self, mask256: torch.Tensor) -> torch.Tensor:
        if tuple(mask256.shape[1:]) != (1, MASK_CANVAS, MASK_CANVAS):
            raise RuntimeError(f"MaskLatentEncoder expects [B,1,256,256], got {tuple(mask256.shape)}")
        value = self.project(self.down64(self.down128(self.stem(mask256))))
        # Every latent leaving Stage 1 lives on the same bounded manifold.
        return normalize_latent(value)


class MaskLatentDecoder(nn.Module):
    """Image-independent latent -> 256x256 mask logits.

    This is the module that makes Stage-2 Phase A meaningful: it converts a
    latent into real geometry without touching SAM's ViT-H image encoder, so
    the alignment phase can be supervised with BCE/Dice at negligible cost.
    """

    def __init__(self, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(int(latent_channels), 128, 1, bias=False), LayerNorm2d(128), nn.GELU(),
            ResidualBlock(128), ResidualBlock(128),
        )
        self.up64 = nn.Sequential(nn.Conv2d(128, 96, 3, padding=1, bias=False), LayerNorm2d(96), nn.GELU(), ResidualBlock(96))
        self.up128 = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1, bias=False), LayerNorm2d(64), nn.GELU(), ResidualBlock(64))
        self.up256 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False), LayerNorm2d(32), nn.GELU(), ResidualBlock(32))
        self.output = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _upsample(value: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(value.float(), size=(size, size), mode="bilinear", align_corners=False).to(dtype=value.dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = self.stem(latent)
        value = self.up64(self._upsample(value, 64))
        value = self.up128(self._upsample(value, 128))
        value = self.up256(self._upsample(value, MASK_CANVAS))
        return self.output(value)


class DensePromptAdapter(nn.Module):
    """Normalized latent -> SAM's native 256x64x64 dense prompt residual."""

    def __init__(self, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.latent_channels, 256, 1, bias=False), LayerNorm2d(256), nn.GELU(),
            ResidualBlock(256), ResidualBlock(256),
        )
        self.refine64 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.output = nn.Conv2d(256, 256, 1)
        self.delta_scale = nn.Parameter(torch.tensor(0.10))
        # Start from SAM's native no-mask prompt so an untrained adapter cannot
        # corrupt the pretrained Two-Way Transformer.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape[1:]) != (self.latent_channels, LATENT_GRID, LATENT_GRID):
            raise RuntimeError(
                f"Dense Prompt Adapter expects [B,{self.latent_channels},{LATENT_GRID},{LATENT_GRID}], "
                f"got {tuple(latent.shape)}"
            )
        value = self.stem(latent)
        value = F.interpolate(value.float(), size=(64, 64), mode="bilinear", align_corners=False).to(dtype=latent.dtype)
        value = self.refine64(value)
        return self.delta_scale.clamp(0.0, 0.5) * torch.tanh(self.output(value))


class MaskLatentTeacher(nn.Module):
    """Stage-1: mask -> normalized latent -> {image-free mask head, SAM decoder}."""

    FORMAT = "mask_latent_teacher_v2_normalized"

    def __init__(self, sam_path: str, sam_model_type: str = "vit_h", latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        if sam_model_type not in sam_model_registry:
            raise ValueError(f"Unsupported SAM model type: {sam_model_type}")
        self.latent_channels = int(latent_channels)
        self.mask_encoder = MaskLatentEncoder(self.latent_channels)
        self.mask_head = MaskLatentDecoder(self.latent_channels)
        self.prompt_adapter = DensePromptAdapter(self.latent_channels)
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_path)
        self.sam.image_encoder.to(dtype=torch.bfloat16)
        self.sam.mask_decoder.to(dtype=torch.bfloat16)
        self.sam.image_encoder.requires_grad_(False)
        self.sam.prompt_encoder.requires_grad_(False)
        self.noise_sigma = 0.0
        self.sam_bce_weight, self.sam_dice_weight = 1.0, 1.0
        self.head_bce_weight, self.head_dice_weight = 1.0, 1.0
        self.head_clean_weight = 0.5
        self.tune_mode = "warmup"
        self.configure_tuning("warmup")
        self._gradient_guarded_parameter_names = set()

    # ------------------------------------------------------------------ data

    @staticmethod
    def _mask_canvas(mask: torch.Tensor, input_size: Sequence[int]) -> torch.Tensor:
        height, width = int(input_size[0]), int(input_size[1])
        resized = F.interpolate(mask.float(), size=(height, width), mode="nearest")
        canvas = F.pad(resized, (0, 1024 - width, 0, 1024 - height))
        return F.interpolate(canvas, size=(MASK_CANVAS, MASK_CANVAS), mode="area")

    def masks256(self, gt_masks: Sequence[torch.Tensor], sam_input_sizes: torch.Tensor) -> torch.Tensor:
        values = [self._mask_canvas(mask.unsqueeze(0), sam_input_sizes[index]) for index, mask in enumerate(gt_masks)]
        parameter = next(self.mask_encoder.parameters())
        return torch.cat(values, dim=0).to(device=parameter.device, dtype=parameter.dtype)

    def encode_gt(self, gt_masks: Sequence[torch.Tensor], sam_input_sizes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask256 = self.masks256(gt_masks, sam_input_sizes)
        return self.mask_encoder(mask256), mask256.float()

    def encode_sam_images(self, sam_images: Sequence[torch.Tensor]) -> torch.Tensor:
        parameter = next(self.sam.image_encoder.parameters())
        images = [self.sam.preprocess(image.to(device=parameter.device, dtype=torch.float32)) for image in sam_images]
        with torch.no_grad():
            return self.sam.image_encoder(torch.stack(images, dim=0).to(dtype=parameter.dtype))

    # -------------------------------------------------------------- decoding

    def decode_mask_head(self, latent: torch.Tensor) -> torch.Tensor:
        head_dtype = next(self.mask_head.parameters()).dtype
        return self.mask_head(latent.to(dtype=head_dtype)).float()

    def prompt_delta(self, latent: torch.Tensor) -> torch.Tensor:
        adapter_dtype = next(self.prompt_adapter.parameters()).dtype
        return self.prompt_adapter(latent.to(dtype=adapter_dtype))

    @property
    def decoder_dtype(self) -> torch.dtype:
        return next(self.sam.mask_decoder.parameters()).dtype

    def decode_prompt(self, prompt_delta: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        """One dense prompt per image, decoded with SAM's single-image semantics."""
        if tuple(prompt_delta.shape[1:]) != (256, 64, 64):
            raise RuntimeError(f"Dense prompt delta must be [B,256,64,64], got {tuple(prompt_delta.shape)}")
        no_mask = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, 256, 1, 1)
        dense_prompt = no_mask.to(device=prompt_delta.device, dtype=prompt_delta.dtype).expand_as(prompt_delta) + prompt_delta
        if dense_prompt.requires_grad:
            dense_prompt.register_hook(sanitize_prompt_gradient)
        decoder_dtype = self.decoder_dtype
        image_pe = self.sam.prompt_encoder.get_dense_pe().to(device=dense_prompt.device, dtype=decoder_dtype)
        masks = []
        for index in range(prompt_delta.shape[0]):
            sparse = torch.empty((1, 0, 256), device=dense_prompt.device, dtype=decoder_dtype)
            mask, _ = self.sam.mask_decoder(
                image_embeddings=sam_features[index:index + 1].to(dtype=decoder_dtype),
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense_prompt[index:index + 1].to(dtype=decoder_dtype),
                multimask_output=False,
            )
            masks.append(mask)
        return torch.cat(masks, dim=0).float()

    def decode_latent(self, latent: torch.Tensor, sam_features: torch.Tensor) -> torch.Tensor:
        return self.decode_prompt(self.prompt_delta(latent), sam_features)

    def original_logits(
        self, masks256: torch.Tensor, sam_input_sizes: torch.Tensor, original_sizes: torch.Tensor,
    ):
        """Restore SAM logits from image geometry only, never from a GT tensor."""
        if masks256.shape[0] != sam_input_sizes.shape[0] or masks256.shape[0] != original_sizes.shape[0]:
            raise RuntimeError("SAM mask batch, input sizes, and original sizes must share a batch dimension.")
        if tuple(original_sizes.shape[1:]) != (2,):
            raise RuntimeError(f"original_sizes must be [B,2] (height,width), got {tuple(original_sizes.shape)}")
        return [
            self.sam.postprocess_masks(
                masks256[index:index + 1],
                tuple(map(int, sam_input_sizes[index])),
                tuple(map(int, original_sizes[index])),
            )
            for index in range(masks256.shape[0])
        ]

    # ------------------------------------------------------------ scheduling

    def set_noise_sigma(self, sigma: float) -> None:
        """Curriculum bound for Stage-1 latent-noise robustness."""
        self.noise_sigma = float(max(0.0, sigma))

    def sample_noise(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.noise_sigma <= 0.0:
            return torch.zeros(batch, device=device)
        # Uniform in [0, sigma_max] keeps clean latents in the batch, so the
        # reconstruction ceiling is preserved while the robust ball grows.
        return torch.rand(batch, device=device) * self.noise_sigma

    def configure_tuning(self, mode: str) -> None:
        if mode not in {"warmup", "full", "freeze", "student_b", "student_b_full"}:
            raise ValueError(
                "tuning mode must be warmup, full, freeze, student_b, or student_b_full"
            )
        self.tune_mode = mode
        for module in (self.mask_encoder, self.mask_head, self.prompt_adapter, self.sam.mask_decoder):
            module.requires_grad_(False)
        if mode == "freeze":
            return
        if mode in {"warmup", "full"}:
            self.mask_encoder.requires_grad_(True)
            self.mask_head.requires_grad_(True)
            self.prompt_adapter.requires_grad_(True)
            if mode == "full":
                self.sam.mask_decoder.requires_grad_(True)
            return
        # student_b/student_b_full: the Prompt Adapter adapts to the Student
        # latent distribution and SAM follows the task loss.  The regular
        # student_b recipe updates only the upper decoder path; student_b_full
        # is an explicit ablation that updates the complete mask decoder while
        # keeping the latent encoder/head frozen.
        self.prompt_adapter.requires_grad_(True)
        if mode == "student_b_full":
            self.sam.mask_decoder.requires_grad_(True)
            return
        transformer = self.sam.mask_decoder.transformer
        transformer.layers[-1].requires_grad_(True)
        transformer.final_attn_token_to_image.requires_grad_(True)
        transformer.norm_final_attn.requires_grad_(True)
        self.sam.mask_decoder.output_upscaling.requires_grad_(True)
        self.sam.mask_decoder.output_hypernetworks_mlps[0].requires_grad_(True)
        self.sam.mask_decoder.mask_tokens.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam.image_encoder.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.train(mode and self.tune_mode in {"full", "student_b", "student_b_full"})
        return self

    def install_trainable_gradient_guards(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and name not in self._gradient_guarded_parameter_names:
                parameter.register_hook(sanitize_parameter_gradient)
                self._gradient_guarded_parameter_names.add(name)

    # --------------------------------------------------------- parameter sets

    def encoder_head_parameter_names(self):
        return {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and (
                name.startswith("mask_encoder.") or name.startswith("mask_head.") or name.startswith("prompt_adapter.")
            )
        }

    def prompt_adapter_parameter_names(self):
        return {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("prompt_adapter.")
        }

    def sam_decoder_parameter_names(self):
        return {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("sam.mask_decoder.")
        }

    # ---------------------------------------------------------------- Stage 1

    def forward(self, sam_images, gt_masks, sam_input_sizes, original_sizes, **_) -> Dict[str, torch.Tensor]:
        latent, mask256 = self.encode_gt(gt_masks, sam_input_sizes)
        sigma = self.sample_noise(latent.shape[0], latent.device)
        noisy = perturb_latent(latent, sigma)

        head_clean = self.decode_mask_head(latent)
        head_noisy = self.decode_mask_head(noisy)
        clean_bce, clean_dice = mask_bce_dice(head_clean, mask256)
        noisy_bce, noisy_dice = mask_bce_dice(head_noisy, mask256)

        sam_features = self.encode_sam_images(sam_images)
        masks256 = self.decode_prompt(self.prompt_delta(noisy), sam_features)
        logits = self.original_logits(masks256, sam_input_sizes, original_sizes)
        terms = [mask_bce_dice(logit, target_for_logits(logit, mask)) for logit, mask in zip(logits, gt_masks)]
        sam_bce = torch.stack([term[0].mean() for term in terms]).mean()
        sam_dice = torch.stack([term[1].mean() for term in terms]).mean()

        loss = (
            self.sam_bce_weight * sam_bce
            + self.sam_dice_weight * sam_dice
            + self.head_bce_weight * noisy_bce.mean()
            + self.head_dice_weight * noisy_dice.mean()
            + self.head_clean_weight * (clean_bce.mean() + clean_dice.mean())
        )
        return {
            "loss": loss,
            "loss_sam_bce": sam_bce, "loss_sam_dice": sam_dice,
            "loss_head_bce": noisy_bce.mean(), "loss_head_dice": noisy_dice.mean(),
            "loss_head_clean_bce": clean_bce.mean(), "loss_head_clean_dice": clean_dice.mean(),
            "noise_sigma": loss.new_tensor(self.noise_sigma),
        }

    # ------------------------------------------------------------ checkpoints

    def save_teacher(self, path: str) -> None:
        state = {}
        for module_name in ("mask_encoder", "mask_head", "prompt_adapter", "sam.mask_decoder"):
            module = self
            for part in module_name.split("."):
                module = getattr(module, part)
            state.update({f"{module_name}.{name}": value.detach().cpu() for name, value in module.state_dict().items()})
        torch.save({
            "format": self.FORMAT,
            "latent_channels": self.latent_channels,
            "latent_grid": LATENT_GRID,
            "readout_grid": READOUT_GRID,
            "noise_sigma": self.noise_sigma,
            "state_dict": state,
        }, path)

    def load_teacher(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != self.FORMAT:
            raise RuntimeError(
                "Not a llava6-3-4 normalized-latent Teacher checkpoint. The llava6-3-1 "
                f"native-16 Teacher cannot be loaded here: {path}"
            )
        if int(checkpoint.get("latent_channels", -1)) != self.latent_channels:
            raise RuntimeError("Teacher latent-channel count does not match this model.")
        if int(checkpoint.get("latent_grid", -1)) != LATENT_GRID:
            raise RuntimeError("Teacher latent grid does not match this model.")
        own = self.state_dict()
        unknown = [name for name in checkpoint["state_dict"] if name not in own]
        if unknown:
            raise RuntimeError(f"Teacher checkpoint has unexpected tensors: {unknown[:4]}")
        for name, value in checkpoint["state_dict"].items():
            own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))






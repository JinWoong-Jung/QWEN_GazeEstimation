from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import compute_total_loss
from .modules import (
    GazeInputResizer,
    GazeRecognitionClassifier,
    HeatmapUpscaler,
    SubjectConditioning,
    SubjectSummary,
)


class QwenGazeIntegratedModel(nn.Module):
    """End-to-end gaze model wrapper around an existing Qwen3-VL backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        scene_grid_size: tuple[int, int] | None,
        *,
        num_classes: int | None = None,
        conditioning_mode: str = "film",
        pool_mode: str = "mean",
        scene_input_size: tuple[int, int] = (512, 512),
        head_input_size: tuple[int, int] = (224, 224),
        heatmap_size: tuple[int, int] = (512, 512),
        num_conditioning_heads: int = 8,
        num_conditioning_layers: int = 1,
        dropout: float = 0.1,
        use_subject_context: bool = True,
        recognition_objective: str = "ce",
        label_emb_dim: int = 512,
        logit_scale_init: float = 0.07,
        lambda_cls: float = 1.0,
        label_smoothing: float = 0.0,
        cls_ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_resizer = GazeInputResizer(
            scene_size=scene_input_size,
            head_size=head_input_size,
        )
        self.summary = SubjectSummary(
            hidden_dim=hidden_dim,
            pool_mode=pool_mode,
            dropout=dropout,
        )
        self.conditioner = SubjectConditioning(
            hidden_dim=hidden_dim,
            mode=conditioning_mode,
            num_heads=num_conditioning_heads,
            dropout=dropout,
            use_pair_tokens=True,
            num_layers=num_conditioning_layers,
        )
        self.localizer = HeatmapUpscaler(
            hidden_dim=hidden_dim,
            scene_grid_size=scene_grid_size,
            output_size=heatmap_size,
            upsample_mode="bilinear",
            apply_sigmoid=True,
        )
        self.recognition_objective = str(recognition_objective).strip().lower()
        self.label_emb_dim = int(label_emb_dim)
        build_recognition = False
        if self.recognition_objective in {"infonce", "batch_local_infonce"}:
            build_recognition = True
        elif self.recognition_objective == "ce" and num_classes is not None and int(num_classes) > 0:
            build_recognition = True
        self.classifier = (
            GazeRecognitionClassifier(
                hidden_dim=hidden_dim,
                num_classes=(int(num_classes) if self.recognition_objective == "ce" else None),
                output_dim=(int(hidden_dim) if self.recognition_objective == "ce" else int(label_emb_dim)),
                mlp_hidden_dim=(
                    int(hidden_dim)
                    if self.recognition_objective in {"infonce", "batch_local_infonce"}
                    else None
                ),
                mlp_num_layers=(
                    6
                    if self.recognition_objective in {"infonce", "batch_local_infonce"}
                    else 2
                ),
                normalize_output=bool(self.recognition_objective in {"infonce", "batch_local_infonce"}),
                dropout=dropout,
                use_subject_context=use_subject_context,
            )
            if build_recognition
            else None
        )
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / max(float(logit_scale_init), 1e-6), dtype=torch.float32))
        )
        self.register_buffer("vocab_emb", torch.empty(0), persistent=False)

        self.lambda_cls = float(lambda_cls)
        self.label_smoothing = float(label_smoothing)
        self.cls_ignore_index = int(cls_ignore_index)

    def set_vocab_embeddings(self, vocab_emb: torch.Tensor | None) -> None:
        if vocab_emb is None:
            self.vocab_emb = torch.empty(0, device=self.logit_scale.device)
            return
        x = vocab_emb.detach().to(device=self.logit_scale.device, dtype=torch.float32)
        if x.dim() != 2:
            raise ValueError(f"vocab_emb must be [V, D], got shape={tuple(x.shape)}")
        self.vocab_emb = F.normalize(x, p=2, dim=-1)

    def _extract_hidden_triplet(self, backbone_output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(backbone_output, (tuple, list)) and len(backbone_output) == 3:
            return backbone_output[0], backbone_output[1], backbone_output[2]
        if isinstance(backbone_output, dict):
            if all(k in backbone_output for k in ("H_s", "H_h", "H_t")):
                return backbone_output["H_s"], backbone_output["H_h"], backbone_output["H_t"]
            if all(k in backbone_output for k in ("scene_hidden", "head_hidden", "text_hidden")):
                return (
                    backbone_output["scene_hidden"],
                    backbone_output["head_hidden"],
                    backbone_output["text_hidden"],
                )
        raise ValueError(
            "Backbone output format is unsupported. "
            "Expected (H_s, H_h, H_t) tuple/list or dict with keys "
            "['H_s','H_h','H_t'] or ['scene_hidden','head_hidden','text_hidden']."
        )

    def _encode_backbone(
        self,
        scene_image: Any,
        head_image: Any,
        text_inputs: Any,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = dict(backbone_kwargs or {})
        if hasattr(self.backbone, "encode") and callable(getattr(self.backbone, "encode")):
            out = self.backbone.encode(scene_image=scene_image, head_image=head_image, text_inputs=text_inputs, **kwargs)
        else:
            out = self.backbone(scene_image=scene_image, head_image=head_image, text_inputs=text_inputs, **kwargs)
        return self._extract_hidden_triplet(out)

    @staticmethod
    def _match_heatmap_shape(
        pred_heatmap: torch.Tensor,
        target_heatmap: torch.Tensor,
    ) -> torch.Tensor:
        tgt = target_heatmap
        if tgt.dim() == 3 and pred_heatmap.dim() == 4 and pred_heatmap.shape[1] == 1:
            tgt = tgt.unsqueeze(1)
        if tgt.shape != pred_heatmap.shape:
            raise ValueError(
                f"target_heatmap shape mismatch: pred={tuple(pred_heatmap.shape)} "
                f"target={tuple(tgt.shape)}"
            )
        return tgt.to(dtype=pred_heatmap.dtype, device=pred_heatmap.device)

    def forward(
        self,
        scene_image: Any,
        head_image: Any,
        text_inputs: Any,
        *,
        target_heatmap: torch.Tensor | None = None,
        target_point: torch.Tensor | None = None,
        target_label: torch.Tensor | None = None,
        target_label_emb: torch.Tensor | None = None,
        target_label_valid: torch.Tensor | None = None,
        use_softargmax: bool | None = None,
        backbone_kwargs: dict[str, Any] | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Any]:
        scene_image, head_image = self.input_resizer(scene_image, head_image)
        h_s, h_h, h_t = self._encode_backbone(
            scene_image=scene_image,
            head_image=head_image,
            text_inputs=text_inputs,
            backbone_kwargs=backbone_kwargs,
        )

        summary_out = self.summary(head_hidden=h_h, text_hidden=h_t)
        cond_out = self.conditioner(
            scene_hidden=h_s,
            subject_token=summary_out["z"],
            head_token=summary_out["z_h"],
            text_token=summary_out["z_t"],
        )
        scene_grid_hw = getattr(self.backbone, "last_scene_grid_hw", None)
        loc_out = self.localizer(
            scene_hidden=cond_out["scene_hidden"],
            scene_grid_size=scene_grid_hw,
            use_softargmax=use_softargmax,
        )

        out: dict[str, Any] = {
            "heatmap": loc_out["heatmap"],
            "heatmap_logits": loc_out["heatmap_logits"],
            "patch_logits": loc_out["patch_logits"],
            "point": loc_out["point"],
            "point_soft": loc_out["point_soft"],
            "point_hard": loc_out["point_hard"],
            "scene_grid_hw": loc_out.get("scene_grid_hw"),
            "z_h": summary_out["z_h"],
            "z_t": summary_out["z_t"],
            "z": summary_out["z"],
        }

        if self.classifier is not None:
            cls_out = self.classifier(
                scene_hidden=cond_out["scene_hidden"],
                patch_logits=loc_out["patch_logits"],
                head_token=summary_out["z_h"],
                text_token=summary_out["z_t"],
            )
            pred_emb = cls_out["f_r"]
            logits = cls_out.get("logits", None)
            pred_label = cls_out.get("pred", None)
            if (
                self.recognition_objective in {"infonce", "batch_local_infonce"}
                and self.vocab_emb.numel() > 0
                and pred_emb.dim() == 2
                and self.vocab_emb.shape[1] == pred_emb.shape[1]
            ):
                logits = torch.matmul(
                    pred_emb.to(dtype=self.vocab_emb.dtype, device=self.vocab_emb.device),
                    self.vocab_emb.t(),
                ) * self.logit_scale.exp()
                pred_label = torch.argmax(logits, dim=-1)
            out.update(
                {
                    "f_g": cls_out["f_g"],
                    "f_r": pred_emb,
                }
            )
            if logits is not None:
                out["logits"] = logits
            if pred_label is not None:
                out["pred_label"] = pred_label

        if return_hidden:
            out["scene_hidden"] = cond_out["scene_hidden"]
            out["head_hidden"] = h_h
            out["text_hidden"] = h_t

        if target_heatmap is not None:
            matched_heatmap = self._match_heatmap_shape(out["heatmap"], target_heatmap)
            loss_out = compute_total_loss(
                pred_heatmap=out["heatmap"],
                target_heatmap=matched_heatmap,
                pred_heatmap_logits=out["heatmap_logits"],
                logits=out.get("logits"),
                target_label=target_label,
                pred_label_emb=out.get("f_r"),
                target_label_emb=target_label_emb,
                target_label_valid=target_label_valid,
                lambda_cls=self.lambda_cls,
                label_smoothing=self.label_smoothing,
                cls_ignore_index=self.cls_ignore_index,
                recognition_objective=self.recognition_objective,
                logit_scale=self.logit_scale,
            )
            out["loss"] = loss_out["loss"]
            out["loss_dict"] = loss_out

        return out

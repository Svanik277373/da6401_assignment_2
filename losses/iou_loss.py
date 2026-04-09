"""Custom IoU loss."""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        centers = boxes[..., :2]
        sizes = boxes[..., 2:].clamp(min=0.0)
        half = sizes * 0.5
        top_left = centers - half
        bottom_right = centers + half
        return torch.cat([top_left, bottom_right], dim=-1)

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute 1 - IoU for normalized center-format boxes."""
        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)

        inter_top_left = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
        inter_bottom_right = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
        inter_sizes = (inter_bottom_right - inter_top_left).clamp(min=0.0)
        inter_area = inter_sizes[..., 0] * inter_sizes[..., 1]

        pred_sizes = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp(min=0.0)
        target_sizes = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp(min=0.0)
        pred_area = pred_sizes[..., 0] * pred_sizes[..., 1]
        target_area = target_sizes[..., 0] * target_sizes[..., 1]

        union = pred_area + target_area - inter_area
        iou = inter_area / (union + self.eps)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

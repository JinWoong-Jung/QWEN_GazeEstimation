from .localization import compute_localization_loss, heatmap_bce_loss
from .recognition import classification_ce_loss, info_nce_batch_local_loss
from .total_loss import compute_total_loss

__all__ = [
    "heatmap_bce_loss",
    "compute_localization_loss",
    "classification_ce_loss",
    "info_nce_batch_local_loss",
    "compute_total_loss",
]

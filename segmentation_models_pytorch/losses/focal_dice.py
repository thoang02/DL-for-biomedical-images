from typing import Optional, List
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ._functional import focal_loss_with_logits
from ._functional import soft_dice_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from . import base
from . import functional as F_dice
from ..base.modules import Activation

__all__ = ["FocalDiceLoss"]

class FocalDiceLoss(_Loss):

    def __init__(
        self,
        mode: str,
        tradeoff: 0.5,
        # focal loss
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.,
        ignore_index: Optional[int] = None, 
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        # dice loss
        eps=1., beta=1., activation=None, ignore_channels=None, **kwargs
      ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.tradeoff = tradeoff

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred_dice_loss = self.activation(y_pred)
        y_true_dice_loss = y_true

        dice_loss = 1 - F_dice.jaccard(
            y_pred_dice_loss, y_true_dice_loss,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            
            num_classes = y_pred.size(1)

            # dice loss

            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                #cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]
                cls_y_true = y_true[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss = loss + self.tradeoff*self.focal_loss_fn(cls_y_pred, cls_y_true) + (1-self.tradeoff)*dice_loss

        return loss

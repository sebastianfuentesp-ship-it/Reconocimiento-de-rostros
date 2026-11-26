"""Sparse center loss with an explicit update for the class-center buffers.

The authors' custom backward returns a center update without multiplying it by
the upstream loss gradient. Applying SGD to those centers at learning rate alpha
therefore implements equation 11 independently of lambda and batch averaging.
Here ordinary autograd differentiates the loss with respect to features and
attention; ``update_centers`` implements that separate update under no_grad.

Reference: https://github.com/amirhfarzaneh/dacl/blob/
c41e9e81de9b70a84cae3d7c19102b5095964d5b/loss.py
"""

import math

import torch
from torch import nn


class SparseCenterLoss(nn.Module):
    """Attention-weighted center distances; centers are state, not parameters."""

    def __init__(self, num_classes: int = 7, feat_dim: int = 512):
        super().__init__()
        if (not isinstance(num_classes, int) or isinstance(num_classes, bool)
                or not isinstance(feat_dim, int) or isinstance(feat_dim, bool)
                or num_classes < 1 or feat_dim < 1):
            raise ValueError("num_classes y feat_dim deben ser enteros positivos.")
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_buffer("centers", torch.empty(num_classes, feat_dim))
        # Match the original implementation's initialization of the transposed buffer.
        nn.init.kaiming_normal_(self.centers.t())

    def _validate(self, features, attention, labels):
        if not all(isinstance(value, torch.Tensor) for value in (features, attention, labels)):
            raise ValueError("features, attention y labels deben ser tensores.")
        if features.ndim != 2 or features.shape[1] != self.feat_dim or features.shape[0] < 1:
            raise ValueError(f"features debe tener forma (N, {self.feat_dim}), con N >= 1.")
        if attention.shape != features.shape:
            raise ValueError("attention debe tener la misma forma (N, D) que features.")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels debe tener forma (N,), con el mismo N que features.")
        if not features.is_floating_point() or not attention.is_floating_point():
            raise ValueError("features y attention deben ser tensores de punto flotante.")
        integer_types = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
        if labels.dtype not in integer_types:
            raise ValueError("labels debe usar un tipo entero de etiquetas.")
        if any(value.device != self.centers.device for value in (features, attention, labels)):
            raise ValueError("features, attention, labels y centers deben estar en el mismo dispositivo.")
        if not all(torch.isfinite(value).all().item() for value in (features, attention, self.centers)):
            raise ValueError("features, attention y centers deben contener solo valores finitos.")
        if ((labels < 0) | (labels >= self.num_classes)).any().item():
            raise ValueError(f"labels debe estar en el intervalo [0, {self.num_classes - 1}].")
        return labels.to(dtype=torch.long)

    def forward(self, features, attention, labels):
        """Return sum(a * (f - c_y)**2) / (2*N) without center gradients."""
        indices = self._validate(features, attention, labels)
        differences = features - self.centers.index_select(0, indices)
        return (attention * differences.square()).sum() / (2 * features.shape[0])

    @torch.no_grad()
    def update_centers(self, features, attention, labels, alpha: float = 0.5):
        """Apply equation 11 using the batch features/attention from before SGD.

        Call once per training batch, after the network optimizer step, with
        detached values from that batch's forward pass. No lambda, momentum,
        weight decay or division by the total batch size enters this update.
        """
        if (not isinstance(alpha, (int, float)) or isinstance(alpha, bool)
                or not math.isfinite(alpha) or not 0 < alpha <= 1):
            raise ValueError("alpha debe ser finito y estar en el intervalo (0, 1].")
        indices = self._validate(features, attention, labels)
        features = features.detach()
        attention = attention.detach()
        differences = self.centers.index_select(0, indices) - features
        weighted_differences = attention * differences
        sums = torch.zeros_like(self.centers, dtype=weighted_differences.dtype)
        sums.index_add_(0, indices, weighted_differences)
        counts = torch.bincount(indices, minlength=self.num_classes).to(dtype=sums.dtype)
        updates = alpha * sums / (1 + counts[:, None])
        self.centers.sub_(updates.to(dtype=self.centers.dtype))

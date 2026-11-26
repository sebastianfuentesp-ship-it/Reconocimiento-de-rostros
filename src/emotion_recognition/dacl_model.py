"""ResNet18 with the auxiliary attention architecture published for DACL.

Architecture reference: amirhfarzaneh/dacl, commit
c41e9e81de9b70a84cae3d7c19102b5095964d5b, models/resnet.py.
Attention weights are used by the training loss, not by the classifier at inference.
"""

from pathlib import Path

import torch
from torch import nn


def _initialize_attention_linear(module: nn.Module) -> None:
    """Initialize only newly added attention linears; leave the backbone intact."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DACLResNet18(nn.Module):
    """Keep the pretrained backbone and attach the original 512 attention heads."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.initialization = backbone.initialization
        self.attention = nn.Sequential(
            nn.Linear(512 * 7 * 7, 3584),
            nn.BatchNorm1d(3584),
            nn.ReLU(inplace=True),
            nn.Linear(3584, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
        )
        self.attention_heads = nn.Linear(64, 512 * 2)
        self.attention.apply(_initialize_attention_linear)
        self.attention_heads.apply(_initialize_attention_linear)

    def _feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return self.backbone.layer4(x)

    def _classify_feature_maps(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.backbone.avgpool(maps), 1)
        return features, self.backbone.fc(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return seven raw logits without evaluating the auxiliary attention branch."""
        _, logits = self._classify_feature_maps(self._feature_maps(images))
        return logits

    def forward_with_attention(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pooled features, logits and channel attention for the DACL loss."""
        maps = self._feature_maps(images)
        if tuple(maps.shape[1:]) != (512, 7, 7):
            raise ValueError("La atención DACL requiere mapas 512x7x7: usa imágenes de 224x224.")
        embeddings = self.attention(torch.flatten(maps, 1))
        attention = self.attention_heads(embeddings).reshape(-1, 512, 2).softmax(dim=-1)[:, :, 1]
        features, logits = self._classify_feature_maps(maps)
        return features, logits, attention


def build_dacl_model(
    pretrained: bool = True, pretrained_weights: str | Path | None = None,
) -> DACLResNet18:
    """Create DACL with face initialization, or offline for checkpoint restoration."""
    # Import locally so the shared checkpoint loader can select this architecture.
    from .model import build_model

    if pretrained:
        backbone = build_model(
            pretrained=True, pretrained_source="msceleb_dacl", pretrained_weights=pretrained_weights,
        )
    else:
        backbone = build_model(pretrained=False)
    return DACLResNet18(backbone)

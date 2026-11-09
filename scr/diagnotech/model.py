"""DenseNet121 multi-label classifier with staged unfreezing and differential learning rates.

Architecture (as specified):
    Input -> DenseNet121 features -> global average pooling (built into torchvision's
    DenseNet forward) -> Dropout(p) -> Linear(1024, 8) -> sigmoid per class (in the loss).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

# Named sub-modules of torchvision's DenseNet121.features, unfrozen cumulatively.
# Level 0 = head only (feature extraction). Level 1 = last dense block. Level 2 = one more block.
UNFREEZE_LEVEL_MODULES: dict[int, tuple[str, ...]] = {
    1: ("features.denseblock4", "features.norm5"),
    2: ("features.transition3", "features.denseblock3"),
}
MAX_UNFREEZE_LEVEL: int = max(UNFREEZE_LEVEL_MODULES)


def build_densenet121(num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """Return a DenseNet121 whose classifier emits ``num_classes`` independent logits."""
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features  # 1024 for DenseNet121
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_unfreeze_level(model: nn.Module, level: int) -> list[str]:
    """Freeze everything, then unfreeze the head and all modules up to ``level``. Returns unfrozen prefixes."""
    if not 0 <= level <= MAX_UNFREEZE_LEVEL:
        raise ValueError(f"unfreeze level must be in [0, {MAX_UNFREEZE_LEVEL}], got {level}")

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.classifier.parameters():
        param.requires_grad_(True)

    unfrozen: list[str] = []
    for lvl in range(1, level + 1):
        unfrozen.extend(UNFREEZE_LEVEL_MODULES[lvl])
    for prefix in unfrozen:
        for param in model.get_submodule(prefix).parameters():
            param.requires_grad_(True)
    return unfrozen


def set_train_mode(model: nn.Module) -> None:
    """Put the model in train mode but keep frozen BatchNorm layers in eval mode.

    Frozen blocks must not update BatchNorm running statistics. Otherwise the
    pretrained ImageNet statistics drift toward this dataset during feature extraction.
    """
    model.train()
    for child in model.features.children():
        if not any(param.requires_grad for param in child.parameters()):
            child.eval()


def build_optimizer(
    model: nn.Module,
    lr_head: float,
    lr_backbone: float | None,
    weight_decay: float,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    """Create an optimizer with separate head and backbone parameter groups.

    The backbone group appears only when some backbone layers are trainable.
    Group order is deterministic, which makes optimizer state loadable on resume.
    """
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]

    groups: list[dict] = [{"name": "head", "params": head_params, "lr": lr_head}]
    if backbone_params:
        if lr_backbone is None:
            raise ValueError("lr_backbone is required when backbone layers are trainable")
        groups.append({"name": "backbone", "params": backbone_params, "lr": lr_backbone})

    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(groups, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}

"""Phase 3 - CNN architectures (PyTorch).  Imported lazily; the rest of the package
works without torch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Binary classifier with a single logit (>0 => fake)."""
    arch = arch.lower()
    if arch in ("resnet18", "resnet50", "resnet152"):
        weights = "DEFAULT" if pretrained else None
        try:
            m = getattr(torchvision.models, arch)(weights=weights)
        except Exception as e:                              # offline: no weight download
            print(f"[models] could not load pretrained {arch} weights ({e}); using random init")
            m = getattr(torchvision.models, arch)(weights=None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
        return m
    if arch == "xception":
        import timm                                          # XceptionNet backbone
        try:
            m = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
        except Exception as e:
            print(f"[models] could not load pretrained xception ({e}); using random init")
            m = timm.create_model("legacy_xception", pretrained=False, num_classes=0)
        return nn.Sequential(m, nn.Dropout(dropout), nn.Linear(m.num_features, 1))
    raise ValueError(f"unknown architecture {arch!r}")


def save_checkpoint(path, model, arch, input_size, extra=None):
    torch.save({"arch": arch, "input_size": input_size, "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck

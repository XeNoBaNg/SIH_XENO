# models/backbones/hybrid_vit_swin.py

from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image

DEFAULT_TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class HybridViTSwinBackbone(nn.Module):
    """
    SIMPLIFIED VERSION: Uses a single, powerful Swin Transformer backbone.
    """
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: Optional[int] = None,
        device: Optional[torch.device] = None,
        pretrained: bool = True,
        **kwargs 
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_embed_dim = self.backbone.num_features

        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(backbone_embed_dim, num_classes)

        self.to(self.device)
        print(f"--- Simplified Backbone Initialized: {model_name} on {self.device} ---")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.classifier:
            return self.classifier(features)
        return features

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt, strict=False)
        print(f"Loaded backbone weights from {path}")
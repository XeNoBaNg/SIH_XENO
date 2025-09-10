<<<<<<< Updated upstream
import torch
import torch.nn as nn

# Optional: when we implement real model, we'll use timm
# import timm

class HybridViTSwinBackbone(nn.Module):
    """
    Stub implementation of a Hybrid ViT + Swin Transformer backbone.
    For now, it just returns dummy feature tensors.
    Later, we will load real pretrained ViT and Swin from timm,
    and fuse their embeddings.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

        # For now, a dummy linear layer to simulate feature extraction
        self.dummy_extractor = nn.Linear(16, embed_dim)

        # CPU by default
        self.device = torch.device("cpu")

        # Deployment (GPU):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Simulate feature extraction.
        x: placeholder (in real case, a tensor image batch)
        """
        if isinstance(x, str):
            # if someone passes a path instead of a tensor
            # just simulate a random tensor
            dummy_input = torch.randn(1, 16).to(self.device)
        elif isinstance(x, torch.Tensor):
            dummy_input = torch.randn(x.size(0), 16).to(self.device)
        else:
            dummy_input = torch.randn(1, 16).to(self.device)

        features = self.dummy_extractor(dummy_input)
        return features

if __name__ == "__main__":
    # Quick self-test
    model = HybridViTSwinBackbone()
    out = model("test.jpg")
    print("Output feature shape:", out.shape)
=======
# models/backbones/hybrid_vit_swin.py

from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image

DEFAULT_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class HybridViTSwinBackbone(nn.Module):
    """
    SIMPLIFIED VERSION: Uses a single, powerful Swin Transformer backbone.
    This removes the complexity and bugs from the hybrid approach.
    """
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: Optional[int] = None,
        device: Optional[torch.device] = None,
        pretrained: bool = True,
        # embed_dim is no longer needed as we use the model's native dimension
        **kwargs 
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a single backbone and remove its final classification layer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the feature dimension directly from the backbone
        backbone_embed_dim = self.backbone.num_features
        
        self.classifier = None
        if num_classes:
            # The classifier directly uses the backbone's output dimension
            self.classifier = nn.Linear(backbone_embed_dim, num_classes)
            
        self.to(self.device)
        print(f"--- Simplified Backbone Initialized: {model_name} on {self.device} ---")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A much simpler and more reliable forward pass."""
        # The timm model with num_classes=0 directly returns features of shape [B, D]
        features = self.backbone(x)
        
        if self.classifier:
            return self.classifier(features)
        return features

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt, strict=False)
        print(f"Loaded backbone weights from {path}")
>>>>>>> Stashed changes

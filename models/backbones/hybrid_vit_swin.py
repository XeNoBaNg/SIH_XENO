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

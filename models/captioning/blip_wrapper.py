"""
BLIP caption wrapper using transformers (Salesforce/blip-image-captioning-base).
Generates a short descriptive caption for civic images.
"""
from typing import Optional
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Assuming you have these utils, otherwise adjust as needed
# from ..utils import get_device, setup_logger
# logger = setup_logger(__name__)

class BLIPCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[torch.device]=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"BLIPCaptioner loaded {model_name} on {self.device}")

    @torch.no_grad()
    def caption(self, image: Image.Image, max_length: int = 50) -> str:
        """Generates a caption for the given PIL image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate the caption
        out = self.model.generate(**inputs, max_length=max_length)
        
        # Decode the output to a string
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption
# models/detectors/yolo_wrapper.py
"""
YOLOv8 wrapper using the ultralytics package.
Provides robust device selection and result normalization.
"""
from typing import List, Dict, Optional, Union
from PIL import Image
import torch

class YOLODetector:
    def __init__(self, weights_path: str = "yolov8n.pt", device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package not found. Please run `pip install ultralytics`.")
        
        self.model = YOLO(weights_path)
        # The ultralytics library handles device mapping well, but this ensures consistency
        self.model.to(self.device)
        print(f"YOLODetector loaded weights={weights_path} on {self.device}")

    def predict(self, image: Union[Image.Image, str], conf: float = 0.25) -> List[Dict]:
        """
        Runs YOLOv8 prediction on a single image.
        Returns a list of dictionaries, each containing 'bbox', 'label', and 'score'.
        """
        # The model from the ultralytics package can accept a PIL Image directly
        results = self.model(image, conf=conf, verbose=False)
        
        output = []
        # The results object is a list; we process the first (and only) result
        for box in results[0].boxes:
            bbox_coords = box.xyxy.tolist()[0]
            label = self.model.names[int(box.cls)]
            score = float(box.conf)
            
            output.append({
                "bbox": [float(coord) for coord in bbox_coords],
                "label": label,
                "score": score
            })
        return output
# models/ensemble/fuser.py
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util

from models.backbones.hybrid_vit_swin import HybridViTSwinBackbone, DEFAULT_TRANSFORMS
from models.detectors.yolo_wrapper import YOLODetector
from models.captioning.blip_wrapper import BLIPCaptioner

CLASS_MAP = {
    0: "Roads and Potholes",
    1: "Streetlight and Electricity",
    2: "Waste Management",
    3: "Water Supply and Sewage"
}


class CivicIssueFuser:
    """
    Final version with the multi-signal spam check.
    """
    def __init__( self, backbone_weights: Optional[str] = None, yolo_weights: str = "yolov8n.pt",
        spam_model: str = 'all-MiniLM-L6-v2',
        device: Optional[torch.device] = None, num_classes: int = 4,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Fuser on device: {self.device}")
        
        self.backbone = HybridViTSwinBackbone(num_classes=num_classes, device=self.device)
        if backbone_weights:
            self.backbone.load_checkpoint(backbone_weights)
        self.backbone.eval()
        
        self.detector = YOLODetector(weights_path=yolo_weights, device=self.device)
        self.captioner = BLIPCaptioner(device=self.device)
        self.spam_checker = SentenceTransformer(spam_model, device=self.device)
# In models/ensemble/fuser.py, inside the CivicIssueFuser class

    @torch.no_grad()
    def predict(self, image: Image.Image, issue_title: str = "", user_description: str = "") -> Dict:
        yolo_predictions = self.detector.predict(image)
        ai_caption = self.captioner.caption(image)
        
        img_tensor = DEFAULT_TRANSFORMS(image).unsqueeze(0).to(self.device)
        backbone_logits = self.backbone(img_tensor)
        final_probs = F.softmax(backbone_logits, dim=1).squeeze()
        prediction_idx = torch.argmax(final_probs).item()
        confidence = final_probs[prediction_idx].item()
        
        predicted_department = CLASS_MAP.get(prediction_idx, "Uncategorized")
        
        # --- Multi-Signal Spam Check Logic (without 'reason') ---
        
        relevance_check = {
            "is_relevant": False,
            "score_visual": 0.0,
            "score_contextual": 0.0
        }
        
        combined_user_text = f"{issue_title}. {user_description}"
        
        if not combined_user_text.strip() or combined_user_text == "N/A":
            relevance_check = {"is_relevant": True, "score_visual": 1.0, "score_contextual": 1.0}
        else:
            user_emb = self.spam_checker.encode(combined_user_text)
            caption_emb = self.spam_checker.encode(ai_caption)
            prediction_emb = self.spam_checker.encode(predicted_department)
            
            score_visual = util.cos_sim(user_emb, caption_emb).item()
            relevance_check["score_visual"] = score_visual
            
            score_contextual = util.cos_sim(user_emb, prediction_emb).item()
            relevance_check["score_contextual"] = score_contextual

            threshold = 0.35
            if score_visual >= threshold and score_contextual >= threshold:
                relevance_check["is_relevant"] = True

        return {
            "department": predicted_department,
            "confidence": confidence,
            "relevance": relevance_check,
            "ai_caption": ai_caption,
            "detections": yolo_predictions
        }
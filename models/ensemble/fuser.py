# models/ensemble/fuser.py
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util

# Import the model wrappers
from models.backbones.hybrid_vit_swin import HybridViTSwinBackbone, DEFAULT_TRANSFORMS
from models.detectors.yolo_wrapper import YOLODetector
from .meta_learner import MetaLearner # Corrected import
from models.captioning.blip_wrapper import BLIPCaptioner

# Define the final classes your system will predict
# models/ensemble/fuser.py



CLASS_MAP = {
    0: "Streetlight and Electricity",
    1: "Waste Management",
    2: "Roads and Potholes",
    3: "waterlogging"
}
# models/ensemble/fuser.py

# (Keep all your existing imports)

# models/ensemble/fuser.py

# (Keep all your existing imports: torch, F, Image, SentenceTransformer, etc.)
# (Also keep your HybridViTSwinBackbone, YOLODetector, etc. imports)
# (Also keep your CLASS_MAP and DEPARTMENT_MAP dictionaries)

class CivicIssueFuser:
    """
    Orchestrates the ML pipeline with department mapping and enhanced spam check.
    """
    def __init__( self, backbone_weights: Optional[str] = None, yolo_weights: str = "yolov8n.pt",
        meta_learner_weights: Optional[str] = None, spam_model: str = 'all-MiniLM-L6-v2',
        device: Optional[torch.device] = None, num_classes: int = 27,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Fuser on device: {self.device}")
        
        self.backbone = HybridViTSwinBackbone(num_classes=num_classes, device=self.device)
        if backbone_weights:
            print(f"Loading custom backbone weights from: {backbone_weights}")
            self.backbone.load_checkpoint(backbone_weights)
        else:
            print("INFO: Using default pre-trained ImageNet weights for the backbone.")
        self.backbone.eval()
        
        self.detector = YOLODetector(weights_path=yolo_weights, device=self.device)
        self.meta_learner = None
        self.captioner = BLIPCaptioner(device=self.device)
        self.spam_checker = SentenceTransformer(spam_model, device=self.device)

    def _is_relevant(self, combined_user_text: str, ai_caption: str, prediction_label: str, threshold: float = 0.50) -> Dict:
        """
        Performs a two-part check using the combined user text.
        """
        if not combined_user_text.strip():
            return {"is_relevant": True, "relevance_score": 1.0, "reason": "No user text provided."}
        
        user_emb = self.spam_checker.encode(combined_user_text, convert_to_tensor=True)
        caption_emb = self.spam_checker.encode(ai_caption, convert_to_tensor=True)
        prediction_emb = self.spam_checker.encode(prediction_label.replace("_", " "), convert_to_tensor=True)
        
        score_vs_caption = util.pytorch_cos_sim(user_emb, caption_emb).item()
        score_vs_prediction = util.pytorch_cos_sim(user_emb, prediction_emb).item()
        
        final_score = max(score_vs_caption, score_vs_prediction)
        is_relevant = final_score >= threshold
        reason = "User input aligns with image content or prediction." if is_relevant else "User input may not match the image content."
        
        return {"is_relevant": is_relevant, "relevance_score": final_score, "reason": reason}

    @torch.no_grad()
    def predict(self, image: Image.Image, issue_title: str = "", user_description: str = "") -> Dict:
        """
        UPDATED to accept 'issue_title'.
        """
        yolo_predictions = self.detector.predict(image)
        ai_caption = self.captioner.caption(image)
        
        img_tensor = DEFAULT_TRANSFORMS(image).unsqueeze(0).to(self.device)
        backbone_logits = self.backbone(img_tensor)
        final_probs = F.softmax(backbone_logits, dim=1).squeeze()
        prediction_idx = torch.argmax(final_probs).item()
        confidence = final_probs[prediction_idx].item()
        
        specific_issue = CLASS_MAP.get(prediction_idx, "unknown_issue")
        main_department = DEPARTMENT_MAP.get(specific_issue, "Uncategorized")
        
        combined_user_text = f"{issue_title}. {user_description}"
        relevance_check = self._is_relevant(combined_user_text, ai_caption, specific_issue)

        return {
            "department": main_department,
            "specific_issue_predicted": specific_issue,
            "confidence": confidence,
            "relevance": relevance_check,
            "ai_caption": ai_caption,
            "detections": yolo_predictions,
            "model_version": "0.9-department-routing"
        }
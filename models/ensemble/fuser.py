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

# Define the final classes your system will predict
CLASS_MAP = {
    0: "broken_streetlight",
    1: "garbage_dump",
    2: "pothole",
    3: "waterlogging"
    # If you train with more classes, add them here in alphabetical order
}

# models/ensemble/fuser.py

# (Keep all your existing imports)

class CivicIssueFuser:
    """
    Orchestrates the ML pipeline with a smarter, two-part spam check.
    """
    def __init__(
        self,
        backbone_weights: Optional[str] = None,
        yolo_weights: str = "yolov8n.pt",
        meta_learner_weights: Optional[str] = None,
        spam_model: str = 'all-MiniLM-L6-v2',
        device: Optional[torch.device] = None,
        num_classes: int = 7,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Fuser on device: {self.device}")
        
        # Load Backbone (using the simplified version)
        self.backbone = HybridViTSwinBackbone(
            num_classes=num_classes,
            device=self.device
        )
        if backbone_weights:
            print(f"Loading custom backbone weights from: {backbone_weights}")
            self.backbone.load_checkpoint(backbone_weights)
        else:
            print("INFO: Using default pre-trained ImageNet weights for the backbone.")
        self.backbone.eval()

        self.detector = YOLODetector(weights_path=yolo_weights, device=self.device)
        self.meta_learner = None # Meta-learner logic can be added later
        
        self.captioner = BLIPCaptioner(device=self.device)
        self.spam_checker = SentenceTransformer(spam_model, device=self.device)

    # --- UPDATED SPAM/RELEVANCE CHECK ---
    def _is_relevant(self, user_desc: str, ai_caption: str, prediction_label: str, threshold: float = 0.50) -> Dict:
        """
        Performs a two-part check for relevance.
        1. Compares user description to the AI caption.
        2. Compares user description to the final model prediction.
        """
        if not user_desc.strip() or user_desc == "N/A":
            return {"is_relevant": True, "relevance_score": 1.0, "reason": "No user description provided."}
        
        # Encode all three pieces of text
        user_emb = self.spam_checker.encode(user_desc, convert_to_tensor=True)
        caption_emb = self.spam_checker.encode(ai_caption, convert_to_tensor=True)
        prediction_emb = self.spam_checker.encode(prediction_label.replace("_", " "), convert_to_tensor=True)
        
        # Check 1: Similarity between user description and AI caption
        score_vs_caption = util.pytorch_cos_sim(user_emb, caption_emb).item()
        
        # Check 2: Similarity between user description and AI prediction
        score_vs_prediction = util.pytorch_cos_sim(user_emb, prediction_emb).item()
        
        # The final score is the HIGHER of the two checks
        final_score = max(score_vs_caption, score_vs_prediction)
        
        is_relevant = final_score >= threshold
        reason = "User description aligns with image content or prediction." if is_relevant else "User description may not match the image content."
        
        return {"is_relevant": is_relevant, "relevance_score": final_score, "reason": reason}

    @torch.no_grad()
    def predict(self, image: Image.Image, user_description: str = "") -> Dict:
        """Runs the full inference pipeline."""
        
        yolo_predictions = self.detector.predict(image)
        ai_caption = self.captioner.caption(image)
        
        img_tensor = DEFAULT_TRANSFORMS(image).unsqueeze(0).to(self.device)
        backbone_logits = self.backbone(img_tensor)
        final_logits = backbone_logits # Using fallback logic

        final_probs = F.softmax(final_logits, dim=1).squeeze()
        prediction_idx = torch.argmax(final_probs).item()
        confidence = final_probs[prediction_idx].item()
        
        # Get the human-readable prediction label
        predicted_label = CLASS_MAP.get(prediction_idx, "unknown")
        
        # Perform the new, smarter relevance check
        relevance_check = self._is_relevant(user_description, ai_caption, predicted_label)

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "relevance": relevance_check,
            "ai_caption": ai_caption,
            "detections": yolo_predictions,
            "model_version": "0.8-smart-spam"
        }
from models.backbones.hybrid_vit_swin import HybridViTSwinBackbone
from models.detectors.deformable_detr import DeformableDetrDetector
from models.detectors.yolo_wrapper import YOLODetector
from models.ensemble.meta_learner import MetaLearner
from models.ensemble.fusion import fuse_predictions
from models.fewshot.prototype_adapter import FewShotAdapter

print("=== Running Step 2 Test ===")

# Init stubs
backbone = HybridViTSwinBackbone()
detr = DeformableDetrDetector()
yolo = YOLODetector()
ensemble = MetaLearner()
fewshot = FewShotAdapter()

# Run Backbone
features = backbone("test.jpg")
print("Backbone output shape:", features.shape)

# Run Detectors
detr_out = detr.predict("test.jpg")
yolo_out = yolo.predict("test.jpg")
print("DETR out:", detr_out)
print("YOLO out:", yolo_out)

# Fuse predictions
fused = fuse_predictions([detr_out, yolo_out])
print("Fused predictions:", fused)

# Ensemble picks final label
labels = [pred["label"] for pred in detr_out + yolo_out]
final = ensemble.predict(labels)
print("Ensemble result:", final)

# Few-shot adapter
fewshot_result = fewshot.predict("test.jpg")
print("FewShot result:", fewshot_result)

print("=== Step 2 Test Completed ===")

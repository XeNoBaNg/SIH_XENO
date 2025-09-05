from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Import our Step 2 stubs
from models.backbones.hybrid_vit_swin import HybridViTSwinBackbone
from models.detectors.deformable_detr import DeformableDetrDetector
from models.detectors.yolo_wrapper import YOLODetector
from models.ensemble.meta_learner import MetaLearner
from models.ensemble.fusion import fuse_predictions
from models.fewshot.prototype_adapter import FewShotAdapter

# Initialize FastAPI app
app = FastAPI(title="Civic Issue Detection API", version="0.1")

# Initialize model stubs (in-memory, on startup)
backbone = HybridViTSwinBackbone()
detr = DeformableDetrDetector()
yolo = YOLODetector()
ensemble = MetaLearner()
fewshot = FewShotAdapter()


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "message": "Civic-AI service is running"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Inference endpoint.
    Accepts an image upload, runs through model pipeline, and returns predictions.
    """

    try:
        # Read image into PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1. Backbone features (not used in stub yet, but extracted anyway)
        features = backbone("dummy_input")

        # 2. Run detectors
        detr_out = detr.predict(image)
        yolo_out = yolo.predict(image)

        # 3. Fuse predictions
        fused = fuse_predictions([detr_out, yolo_out])

        # 4. Meta-learner: pick final label from candidates
        labels = [pred["label"] for pred in detr_out + yolo_out]
        final = ensemble.predict(labels)

        # 5. Few-shot adapter
        fewshot_result = fewshot.predict(image)

        response = {
            "final_label": final["final_label"],
            "confidence": final["confidence"],
            "detections": fused["fused_predictions"],
            "fewshot": fewshot_result["predicted_label"],
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

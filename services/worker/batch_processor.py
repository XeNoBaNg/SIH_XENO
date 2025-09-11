# services/worker/batch_processor.py
import json
import uuid
import base64
import io
import redis
from kafka import KafkaConsumer, KafkaProducer
from PIL import Image
import datetime

# Import the Fuser
from models.ensemble.fuser import CivicIssueFuser

# ---- Config ----
KAFKA_BROKERS = ["localhost:9092"]
REQUEST_TOPIC = "civic-issues"
RESULT_TOPIC = "civic-results"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# ---- Model Paths ----
BACKBONE_WEIGHTS = "weights/backbone.pth"
YOLO_WEIGHTS = "yolov8n.pt" 

# ---- Initialize Fuser and Redis Client ----
print("Initializing Civic Issue Fuser pipeline...")
fuser = CivicIssueFuser(
    backbone_weights=BACKBONE_WEIGHTS,
    yolo_weights=YOLO_WEIGHTS,
    num_classes=4
)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
print("Redis client connected.")


# In services/worker/batch_processor.py

def run_worker():
    consumer = KafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        group_id="civic-fuser-workers",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v, indent=2).encode("utf-8"),
    )

    print("⚡ AI Worker started, listening for inference jobs...")

    for msg in consumer:
        data = msg.value
        request_id = data.get("request_id", str(uuid.uuid4()))
        print(f"➡️  Received job {request_id}")

        try:
            image_b64 = data["image_b64"]
            issue_title = data.get("issue_title", "")
            user_desc = data.get("user_description", "")
            
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            prediction_result = fuser.predict(
                image=image,
                issue_title=issue_title,
                user_description=user_desc
            )

            relevance_data = prediction_result.get("relevance", {})
            spam_status = "Not Spam" if relevance_data.get("is_relevant") else "Potential Spam"
            
            confidence_score = prediction_result.get("confidence", 0)
            if confidence_score > 0.9:
                confidence_level = "Very High"
            elif confidence_score > 0.7:
                confidence_level = "High"
            else:
                confidence_level = "Medium"

            final_result = {
                "request_id": request_id,
                "status": "processed",
                "processed_at": datetime.datetime.now().isoformat(),
                "user_input": {
                    "issue_title": issue_title,
                    "description": user_desc
                },
                "ai_analysis": {
                    "department": prediction_result.get("department"),
                    "confidence": confidence_score,
                    "confidence_level": confidence_level,
                    "spam_check": {
                        "status": spam_status,
                        "visual_score": relevance_data.get("score_visual"),
                        "contextual_score": relevance_data.get("score_contextual")
                        # --- "reason" field removed ---
                    },
                    "image_content": {
                        "ai_caption": prediction_result.get("ai_caption"),
                        "detected_objects": prediction_result.get("detections")
                    }
                },
                "model_version": prediction_result.get("model_version")
            }

        except Exception as e:
            print(f"❌ Error processing job {request_id}: {e}")
            final_result = {
                "request_id": request_id,
                "status": "error",
                "message": str(e),
            }

        producer.send(RESULT_TOPIC, final_result)
        redis_client.setex(f"result:{request_id}", 3600, json.dumps(final_result))
        
        print(f"✅ Processed job {request_id}, results stored.")


if __name__ == "__main__":
    run_worker()
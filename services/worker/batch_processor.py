# services/worker/batch_processor.py
import json
import uuid
<<<<<<< Updated upstream
=======
import base64
import io
import redis
>>>>>>> Stashed changes
from kafka import KafkaConsumer, KafkaProducer

<<<<<<< Updated upstream
# Load Kafka config
=======
# Import the Fuser
from models.ensemble.fuser import CivicIssueFuser

# ---- Config ----
>>>>>>> Stashed changes
KAFKA_BROKERS = ["localhost:9092"]
REQUEST_TOPIC = "civic-issues"
RESULT_TOPIC = "civic-results"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

<<<<<<< Updated upstream
=======
# ---- Model Paths ----
# CORRECTED: Set paths to None since we are using pre-trained defaults
BACKBONE_WEIGHTS = "weights/backbone.pth"
META_LEARNER_WEIGHTS = None

# For YOLO, just provide the model name. The library will handle the download.
YOLO_WEIGHTS = "yolov8n.pt" 

# ---- Initialize the Fuser and Redis Client ONCE at startup ----

print("Initializing Civic Issue Fuser pipeline...")
fuser = CivicIssueFuser(
    backbone_weights=BACKBONE_WEIGHTS,
    yolo_weights=YOLO_WEIGHTS,
    meta_learner_weights=META_LEARNER_WEIGHTS,
    num_classes=4  # <-- ADD THIS LINE
)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
print("Redis client connected.")


>>>>>>> Stashed changes
def run_worker():
    consumer = KafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        group_id="civic-fuser-workers",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest"
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print("⚡ AI Worker started, listening for inference jobs...")

    for msg in consumer:
        data = msg.value
        request_id = data.get("request_id", str(uuid.uuid4()))
<<<<<<< Updated upstream

        print(f"➡️ Received job {request_id}")

        # --- STUB INFERENCE (replace with real model later) ---
        result = {
            "request_id": request_id,
            "final_label": "pothole",
            "confidence": 0.85,
            "detections": [
                {"bbox": [50, 60, 200, 220], "label": "pothole", "score": 0.85}
            ],
            "fewshot": "pothole"
        }

        # Send result to results topic
        producer.send(RESULT_TOPIC, result)
        print(f"✅ Processed job {request_id}, pushed to results topic")
=======
        print(f"➡️  Received job {request_id}")

        try:
            image_b64 = data["image_b64"]
            user_desc = data.get("user_description", "")
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Run the full prediction pipeline
            prediction_result = fuser.predict(image, user_description=user_desc)

            final_result = {
                "request_id": request_id,
                "status": "processed",
                "user_description": user_desc,
                **prediction_result
            }

        except Exception as e:
            print(f"❌ Error processing job {request_id}: {e}")
            final_result = {
                "request_id": request_id,
                "status": "error",
                "message": str(e),
            }

        # 1. Push to Kafka topic for logging or stream processing
        producer.send(RESULT_TOPIC, final_result)
        
        # 2. Store in Redis for fast API retrieval (key expires in 1 hour)
        redis_client.setex(f"result:{request_id}", 3600, json.dumps(final_result))
        
        print(f"✅ Processed job {request_id}, results stored.")


if __name__ == "__main__":
    run_worker()
>>>>>>> Stashed changes

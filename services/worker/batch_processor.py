# services/worker/batch_processor.py
import json
import uuid
from kafka import KafkaConsumer, KafkaProducer

# Load Kafka config
KAFKA_BROKERS = ["localhost:9092"]
REQUEST_TOPIC = "civic-issues"
RESULT_TOPIC = "civic-results"

def run_worker():
    consumer = KafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        group_id="civic-workers",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest"
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print("⚡ Worker started, listening for messages...")

    for msg in consumer:
        data = msg.value
        request_id = data.get("request_id", str(uuid.uuid4()))

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

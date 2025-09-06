from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uuid, json
from kafka import KafkaProducer, KafkaConsumer

# ========================
# Kafka Config
# ========================
KAFKA_BROKERS = ["localhost:9092"]
REQUEST_TOPIC = "civic-issues"
RESULT_TOPIC = "civic-results"

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# ========================
# FastAPI App
# ========================
app = FastAPI(title="Civic Issue Detection API", version="0.3")


# ---- 1. Root ----
@app.get("/")
async def root():
    return {
        "message": "Welcome to Civic-AI API 🚀",
        "docs": "Visit /docs for API documentation and testing."
    }


# ---- 2. Health Check ----
@app.get("/health")
async def health_check():
    return {"status": "success", "message": "Civic-AI service is running"}


# ---- 3. Inference (push job to Kafka) ----
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        request_id = str(uuid.uuid4())

        # For now, we only pass filename (later we will save file / base64 encode)
        job = {
            "request_id": request_id,
            "filename": file.filename
        }

        # Push to Kafka requests topic
        producer.send(REQUEST_TOPIC, job)

        return {"status": "queued", "request_id": request_id}

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


# ---- 4. Fetch Result (consume from results topic) ----
@app.get("/result/{request_id}")
async def get_result(request_id: str):
    try:
        # Create a temporary consumer to fetch results
        consumer = KafkaConsumer(
            RESULT_TOPIC,
            bootstrap_servers=KAFKA_BROKERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            consumer_timeout_ms=2000   # stop after 2 seconds
        )

        for msg in consumer:
            result = msg.value
            if result.get("request_id") == request_id:
                return result

        return {"status": "pending", "request_id": request_id}

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

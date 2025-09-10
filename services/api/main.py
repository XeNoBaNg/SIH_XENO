<<<<<<< Updated upstream
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uuid, json
from kafka import KafkaProducer, KafkaConsumer

# ========================
# Kafka Config
# ========================
=======
# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from kafka import KafkaProducer
import redis
import json
import uuid
import base64

# ---- Config ----
>>>>>>> Stashed changes
KAFKA_BROKERS = ["localhost:9092"]
REQUEST_TOPIC = "civic-issues"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

<<<<<<< Updated upstream
# Initialize Kafka Producer
=======
# ---- Initialize Producers and Clients ----
>>>>>>> Stashed changes
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

<<<<<<< Updated upstream
# ========================
# FastAPI App
# ========================
app = FastAPI(title="Civic Issue Detection API", version="0.3")


# ---- 1. Root ----
@app.get("/")
=======
# ---- FastAPI App ----
app = FastAPI(title="Civic Issue Detection API", version="0.6-hybrid")

@app.get("/", summary="Root Endpoint")
>>>>>>> Stashed changes
async def root():
    return {
        "message": "Welcome to Civic-AI API 🚀",
        "docs": "Visit /docs for API documentation."
    }

<<<<<<< Updated upstream

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
=======
@app.post("/infer", summary="Submit an issue for inference")
async def infer(
    file: UploadFile = File(..., description="Image of the civic issue."),
    description: str = Form("", description="Optional user description of the issue.")
):
    """
    Accepts an image and a description, and queues it for processing by an AI worker.
    Returns a unique `request_id` to track the job.
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image.")
            
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode("utf-8")
        request_id = str(uuid.uuid4())

        job = {
            "request_id": request_id,
            "filename": file.filename,
            "user_description": description,
            "image_b64": image_b64
>>>>>>> Stashed changes
        }

        # Push to Kafka requests topic
        producer.send(REQUEST_TOPIC, job)

        # Set an initial "pending" status in Redis
        initial_status = {"status": "queued", "request_id": request_id}
        redis_client.setex(f"result:{request_id}", 3600, json.dumps(initial_status))

        return JSONResponse(content=initial_status, status_code=202)

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

<<<<<<< Updated upstream

# ---- 4. Fetch Result (consume from results topic) ----
@app.get("/result/{request_id}")
=======
@app.get("/result/{request_id}", summary="Fetch the result of a processed issue")
>>>>>>> Stashed changes
async def get_result(request_id: str):
    """
    Fetches the processing result for a given `request_id`.
    Poll this endpoint until the status is 'processed' or 'error'.
    """
    try:
<<<<<<< Updated upstream
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
=======
        result_json = redis_client.get(f"result:{request_id}")
        
        if result_json:
            return json.loads(result_json)
        else:
            raise HTTPException(status_code=404, detail="Result not found or expired.")
>>>>>>> Stashed changes

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching the result: {e}"
        )
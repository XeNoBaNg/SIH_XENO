# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from kafka import KafkaProducer
import redis
import json
import uuid
import base64

# ---- Config ----
KAFKA_BROKERS = ["localhost:9092"]
REQUEST_TOPIC = "civic-issues"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# ---- Initialize Producers and Clients ----
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# ---- FastAPI App ----
app = FastAPI(title="Civic Issue Detection API", version="1.0")

@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "message": "Welcome to Civic-AI API 🚀",
        "docs": "Visit /docs for API documentation."
    }

@app.post("/infer", summary="Submit an issue for inference")
async def infer(
    file: UploadFile = File(..., description="Image of the civic issue."),
    issue_title: str = Form(..., description="The specific issue title selected by the user."),
    description: str = Form("", description="Optional user-written description of the issue.")
):
    """
    Accepts an image and descriptions, and queues it for processing by an AI worker.
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
            "issue_title": issue_title,
            "user_description": description,
            "image_b64": image_b64
        }
        producer.send(REQUEST_TOPIC, job)
        producer.flush()

        initial_status = {"status": "queued", "request_id": request_id}
        redis_client.setex(f"result:{request_id}", 3600, json.dumps(initial_status))

        return JSONResponse(content=initial_status, status_code=202)

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/result/{request_id}", summary="Fetch the result of a processed issue")
async def get_result(request_id: str):
    """
    Fetches the processing result for a given `request_id`.
    Poll this endpoint until the status is 'processed' or 'error'.
    """
    try:
        result_json = redis_client.get(f"result:{request_id}")

        if result_json:
            return json.loads(result_json)
        else:
            raise HTTPException(status_code=404, detail="Result not found or expired.")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching the result: {e}"
        )
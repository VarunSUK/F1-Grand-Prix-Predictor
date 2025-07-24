from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import predict
import os
from dotenv import load_dotenv
import logging
import json

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "0") == "1"
LOG_FILE = os.getenv("LOG_FILE", "backend.log")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'level': record.levelname,
            'time': self.formatTime(record, self.datefmt),
            'name': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

if LOG_TO_FILE:
    handler = logging.FileHandler(LOG_FILE)
    formatter = JsonFormatter()
else:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] %(message)s')

handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(handler)

app = FastAPI(
    title="F1 Grand Prix Predictor",
    description="ML-powered F1 race winner prediction API",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Set ALLOWED_ORIGINS in .env for production, e.g. "https://yourdomain.com"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "F1 Grand Prix Predictor API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "f1-predictor-api"} 
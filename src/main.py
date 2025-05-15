from fastapi import FastAPI
from api.v1.routes import router as api_router

app = FastAPI(
    title="AI Document Processor",
    version="1.0.0",
    description="Classifies, extracts, and validates documents using Azure/Google AI"
)

app.include_router(api_router, prefix="/api/v1")

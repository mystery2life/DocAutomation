from fastapi import APIRouter, UploadFile, File, Form
from services.dispatcher import route_document
from services.classification.google_docai import classify_document

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    contents = await file.read()
    result = route_document(contents, file.filename)
    return result

@router.post("/classify-document")
async def classify_document_route(
    file: UploadFile = File(...),
    filename: str = Form(...)
):
    file_bytes = await file.read()
    doc_type = classify_document(file_bytes, filename)

    # Return dummy confidence score based on classification
    confidence = 0.95 if doc_type != "unknown" else 0.5

    return {
        "doc_type": doc_type,
        "confidence": confidence
    }


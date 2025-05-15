from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    # This is where classifier → processor → validation will happen
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }

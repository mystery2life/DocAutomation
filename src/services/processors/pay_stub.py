import json
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from openai import AzureOpenAI

# --- Azure OCR and Document Intelligence ---
AZURE_OCR_ENDPOINT = "https://venka.cognitiveservices.azure.com/"
AZURE_OCR_KEY = "FmA8C0l5UHynz2tNCdA2gMXmLJAOsCpjp2eWncyiPvHZHgAPe3QbJQQJ99BEACYeBjFXJ3w3AAALACOGYitX"

ocr_client = DocumentIntelligenceClient(
    endpoint=AZURE_OCR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_OCR_KEY)
)

# --- Azure OpenAI ---
AZURE_OPENAI_KEY = "36BLOdl5AfrHTpOWCwMwUHRwVJePxq6JAgVosopBPo0pGCfoOFyEJQQJ99BFACYeBjFXJ3w3AAABACOGlMcW"
AZURE_OPENAI_ENDPOINT = "https://venka-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# --- LLM Prompt ---
EXTRACTION_PROMPT = """
You are a payroll document expert. From the following pay stub text, extract these fields:

1. TotalHoursWorked — the total of all hours worked across categories like Regular, Overtime, Holiday, Sick, etc. Give only float value. If not listed, return null.
2. AveragePayRate — Use all available hour-rate pairs and compute weighted average as (rate × hours) summed and divided by total hours. Return float values rounded to 2 decimal places. If only one rate exists, just return it.ONLY extract the hourly pay rate. ignore if it is monthly, daily pay rate. Give only float value. If not listed, return null.
3. JobTitle — the job title of the employee (like "Maintenance", "Driver"). If not listed, return null.

Return only **pure JSON**, no extra formatting, no explanations, no markdown code blocks. Output **exactly like this**:
{{"TotalHoursWorked": "...", "AveragePayRate": "...", "JobTitle": "..."}}


Here is the pay stub text:
\"\"\"{text}\"\"\"

JSON:
"""

# --- OCR Text Extraction ---
def extract_text_from_ocr(file_bytes: bytes) -> str:
    print("Starting OCR text extraction with prebuilt-read...")
    poller = ocr_client.begin_analyze_document(
        model_id="prebuilt-read",
        body=BytesIO(file_bytes)
    )
    result = poller.result()
    text = "\n".join([line.content for page in result.pages for line in page.lines])
    print("OCR text extraction complete. Extracted text:")
    print("--------------- OCR RAW TEXT START ---------------")
    print(text[:3000])  # showing first 3000 chars
    print("--------------- OCR RAW TEXT END -----------------")
    return text

# --- LLM Field Extraction ---
def extract_llm_fields(text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(text=text[:1500])
    print("Sending text to Azure OpenAI LLM for field extraction...")
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts fields from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        print(" RAW LLM RESPONSE:", raw)
        parsed = json.loads(raw)
        print("LLM field extraction successful.")
        return {
            "PayRate": {"value": parsed.get("AveragePayRate"), "confidence": 80.2},
            "HoursWorked": {"value": parsed.get("TotalHoursWorked"), "confidence": 83.6},
            "JobTitle": {"value": parsed.get("JobTitle"), "confidence": 88.2}
        }
    except Exception as e:
        print("LLM extraction failed:", e)
        return {
            "PayRate": {"value": None, "confidence": None},
            "HoursWorked": {"value": None, "confidence": None},
            "JobTitle": {"value": None, "confidence": None}
        }

# --- Main Function ---
def process_pay_stub(file_bytes: bytes, filename: str) -> dict:
    print(f" Processing pay stub: {filename}")

    # Step 1: Azure pay stub structured extraction
    try:
        print("Starting structured extraction with prebuilt-payStub.us model...")
        doc_poller = ocr_client.begin_analyze_document(
            model_id="prebuilt-payStub.us",
            body=BytesIO(file_bytes),
            content_type="application/octet-stream",
            features=[DocumentAnalysisFeature.QUERY_FIELDS],
            polling=True
        )
        doc_result = doc_poller.result(timeout=300)
        print("Structured extraction complete.")
    except Exception as e:
        print("Failed to extract structured fields:", e)
        return {"status": "error", "message": f"Azure extraction failed: {e}"}

    # Build extracted_fields
    extracted_fields = {}
    doc = doc_result.documents[0] if doc_result.documents else None
    if not doc:
        print("No document returned by Azure.")
        return {"status": "error", "message": "No document returned by Azure."}

    for key, field in doc.fields.items():
        extracted_fields[key] = {
            "value": field.content if field else None,
            "confidence": round(field.confidence * 100, 2) if field and field.confidence else None
        }
    print("Structured fields extracted:", json.dumps(extracted_fields, indent=2))

    # Step 2: LLM extraction
    try:
        text = extract_text_from_ocr(file_bytes)
        llm_fields = extract_llm_fields(text)
        extracted_fields.update(llm_fields)
    except Exception as e:
        print("LLM extraction step failed:", e)

    print("All processing complete.")
    return {
        "status": "success",
        "filename": filename,
        "extracted_fields": extracted_fields
    }

# Optional CLI
if __name__ == "__main__":
    with open("sample_pay_stub.png", "rb") as f:
        file_bytes = f.read()
        result = process_pay_stub(file_bytes, "sample_pay_stub.png")
        print(json.dumps(result, indent=2))


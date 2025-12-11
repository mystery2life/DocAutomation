# classifier_module/config.py
import os
from dataclasses import dataclass

# Map raw DI doc_type values (normalized) to internal labels
DOC_TYPE_MAP = {
    "paystub": "paystub",
    "employmentverificationform": "ev_form",
    "ev_form": "ev_form",
    "bankstatement": "bank_stmt",
    "bankstatementus": "bank_stmt",
    "bankstatement_us": "bank_stmt",
    "bank statement": "bank_stmt",
    "driverlicense": "dl",
    "driver license": "dl",
    "drivinglicense": "dl",
    "driving license": "dl",
    "dl": "dl",
    "passport": "passport",
    "ssn": "ssn",
}


@dataclass
class Settings:
    az_di_endpoint: str = os.getenv("AZ_DI_ENDPOINT", "").strip()
    az_di_key: str = os.getenv("AZ_DI_KEY", "").strip()
    di_classifier_model_id: str = os.getenv("DI_CLASSIFIER_MODEL_ID", "").strip()

    # confidence threshold for doc-type; below this we force "unknown"
    classifier_conf_threshold: float = float(os.getenv("CLASSIFIER_CONF_THRESHOLD", "0.30"))


settings = Settings()


def _normalize_key(raw: str | None) -> str | None:
    if not raw:
        return None
    return raw.strip().lower().replace(" ", "")


def normalize_doc_type(raw_doc_type: str | None, confidence: float | None) -> str:
    """
    Apply mapping + confidence rule:
    - If raw_doc_type is None -> 'unknown'
    - Map via DOC_TYPE_MAP
    - If confidence < threshold -> force 'unknown'
    """
    # Rule 2: if doc type is None => unknown
    if raw_doc_type is None:
        return "unknown"

    key = _normalize_key(raw_doc_type)
    mapped = DOC_TYPE_MAP.get(key, "unknown")

    # Treat None confidence as 0.0
    conf_value = float(confidence) if confidence is not None else 0.0

    # Rule 3: if confidence < threshold => unknown
    if conf_value < settings.classifier_conf_threshold:
        return "unknown"

    return mapped







# classifier_module/di_client.py
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from .config import settings


def make_di_client() -> DocumentIntelligenceClient:
    """
    Build and return an Azure Document Intelligence client
    for the classifier.
    """
    if not settings.az_di_endpoint or not settings.az_di_key:
        raise RuntimeError("AZ_DI_ENDPOINT or AZ_DI_KEY not configured")

    return DocumentIntelligenceClient(
        endpoint=settings.az_di_endpoint,
        credential=AzureKeyCredential(settings.az_di_key),
        api_version="2024-11-30",
    )




# classifier_module/models.py
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    doc_type: str
    pages: List[int] = Field(..., min_items=1)
    # None means: pixel extraction failed / not available
    # Otherwise: { page_number: [[x1, y1], [x2, y2], ...] }
    doc_pixel_coordinates: Optional[Dict[int, List[List[float]]]] = None
    # Empty now; filled by downstream processors
    result_json: Dict = Field(default_factory=dict)


class ClassificationEnvelope(BaseModel):
    uuid: str
    processed_utc: str
    blob_path: str
    no_of_pages: int
    documents_json_data: Dict[str, DocumentItem]




# classifier_module/pixel_extractor.py
from __future__ import annotations

from typing import Dict, List, Optional


def extract_pixel_coordinates(
    pdf_bytes: bytes,
    pages: list[int],
) -> Optional[Dict[int, List[List[float]]]]:
    """
    Placeholder for your real pixel-extraction logic.

    For now:
    - We do NOT implement any detection.
    - We simply return None to indicate "no pixel coordinates".

    Later you can replace this with:
    - PDF -> image conversion
    - OpenCV-based detection
    - etc.
    """
    try:
        # TODO: implement real extraction here
        return None
    except Exception:
        # Rule 5: ignore pixel extractor module issues and default to None
        return None


# classifier_module/pipeline.py
from __future__ import annotations

import datetime as dt
import io
import logging
from typing import Dict, List, Optional

from azure.core.exceptions import AzureError
from pypdf import PdfReader

from .config import normalize_doc_type, settings
from .di_client import make_di_client
from .models import ClassificationEnvelope, DocumentItem
from .pixel_extractor import extract_pixel_coordinates


class IntermediatePrediction:
    """
    Internal representation of one DI document prediction.
    """

    def __init__(
        self,
        doc_type: str,
        confidence: float,
        pages: List[int],
        doc_pixel_coordinates: Optional[Dict[int, List[List[float]]]],
    ) -> None:
        self.doc_type = doc_type
        self.confidence = confidence
        self.pages = pages
        self.doc_pixel_coordinates = doc_pixel_coordinates


def _utc_timestamp() -> str:
    """Return current UTC time as ISO string without microseconds."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Count pages in the PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


def classify_pdf_with_di(pdf_bytes: bytes) -> List[IntermediatePrediction]:
    """
    Call Azure DI classifier with split='auto' and return a list of IntermediatePrediction.

    Applies:
    - doc_type normalization
    - confidence threshold rule
    - uses DI bounding_regions only to determine pages
    - pixel coordinates are obtained via pixel_extractor (or None on failure)
    """
    client = make_di_client()

    try:
        poller = client.begin_classify_document(
            classifier_id=settings.di_classifier_model_id,
            body=pdf_bytes,
            split="auto",
            content_type="application/pdf",
        )
        result = poller.result(timeout=180)
    except AzureError:
        logging.exception("Azure DI classify_document failed")
        raise

    predictions: List[IntermediatePrediction] = []

    documents = getattr(result, "documents", None) or []
    if not documents:
        # DI returned 0 documents; caller will handle this case specially.
        return predictions

    for doc in documents:
        raw_doc_type = getattr(doc, "doc_type", None)
        confidence = getattr(doc, "confidence", None)

        # Normalize type with confidence rule
        normalized_type = normalize_doc_type(raw_doc_type, confidence)

        # Extract pages from bounding_regions if present
        pages: List[int] = []
        bounding_regions = getattr(doc, "bounding_regions", None)

        if bounding_regions:
            for br in bounding_regions:
                page_num = getattr(br, "page_number", None)
                if page_num is not None:
                    pages.append(int(page_num))

        # Deduplicate and sort pages
        pages = sorted(set(pages))

        # Extremely defensive fallback; you said missing page numbers shouldn't happen
        if not pages:
            logging.warning(
                "DI document has no pages in bounding_regions; defaulting to [1]"
            )
            pages = [1]

        # Pixel coordinates via separate module
        try:
            doc_pixel_coords = extract_pixel_coordinates(pdf_bytes, pages)
        except Exception:
            # Rule 5: ignore pixel extractor issues, default to None
            logging.exception("Pixel extraction failed; setting doc_pixel_coordinates=None")
            doc_pixel_coords = None

        prediction = IntermediatePrediction(
            doc_type=normalized_type,
            confidence=float(confidence or 0.0),
            pages=pages,
            doc_pixel_coordinates=doc_pixel_coords,
        )
        predictions.append(prediction)

    # Sort predictions by first page, just for stability
    predictions.sort(key=lambda p: p.pages[0])

    return predictions


def build_classification_envelope(
    pdf_bytes: bytes,
    blob_path: str,
    job_uuid: str,
    total_pages: Optional[int] = None,
) -> ClassificationEnvelope:
    """
    Main entrypoint for function_app:

    - Uses DI classifier to get document predictions
    - Applies unknown-type rules & confidence rules
    - Uses pixel extractor for doc_pixel_coordinates (or None)
    - Builds the final ClassificationEnvelope with DOC_IDx entries

    Rules from you:
    1) uuid is taken from message (job_uuid), not generated here.
    2) If DI gives 0 docs, we create one unknown doc per page (DOC_ID1..n),
       doc_pixel_coordinates = None.
    3) If raw_doc_type is None => default to 'unknown' (handled in normalize_doc_type()).
    4) If any pixel extractor issue => doc_pixel_coordinates = None.
    """
    # Determine total pages
    if total_pages is not None and total_pages > 0:
        no_of_pages = int(total_pages)
    else:
        no_of_pages = get_pdf_page_count(pdf_bytes)

    # First, try DI classifier
    predictions = classify_pdf_with_di(pdf_bytes)

    documents_json_data: Dict[str, DocumentItem] = {}

    if not predictions:
        # Rule 2: DI gave 0 docs -> one unknown doc per page, pixel coords null
        for page in range(1, no_of_pages + 1):
            doc_id = f"DOC_ID{page}"
            item = DocumentItem(
                doc_type="unknown",
                pages=[page],
                doc_pixel_coordinates=None,
                result_json={},
            )
            documents_json_data[doc_id] = item
    else:
        # Normal path: use DI predictions
        for index, pred in enumerate(predictions, start=1):
            doc_id = f"DOC_ID{index}"
            item = DocumentItem(
                doc_type=pred.doc_type,
                pages=pred.pages,
                doc_pixel_coordinates=pred.doc_pixel_coordinates,
                result_json={},
            )
            documents_json_data[doc_id] = item

    envelope = ClassificationEnvelope(
        uuid=job_uuid,
        processed_utc=_utc_timestamp(),
        blob_path=blob_path,
        no_of_pages=no_of_pages,
        documents_json_data=documents_json_data,
    )

    return envelope


from classifier_module.pipeline import build_classification_envelope

# ...

msg_uuid = json_dict.get("uuid")  # or whatever key you use
page_number = json_dict.get("page_number")

envelope = build_classification_envelope(
    pdf_bytes=pdf_BYTES,
    blob_path=blob_path,
    job_uuid=msg_uuid,
    total_pages=page_number,
)

envelope_dict = envelope.model_dump()
logging.info("Classification envelope: %s", json.dumps(envelope_dict))





pymupdf
opencv-python
numpy





# classifier_module/pixel_extractor.py
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional

import numpy as np
import cv2
import fitz  # PyMuPDF


def _pdf_page_to_image_from_bytes(
    pdf_bytes: bytes,
    page_num: int,
    zoom: float = 2.0,
) -> np.ndarray:
    """
    Render a single PDF page to an OpenCV BGR image.

    Args:
        pdf_bytes: Full PDF file in bytes.
        page_num: 1-based page index.
        zoom: Zoom factor for rendering (higher = more pixels).

    Returns:
        img: np.ndarray (H, W, 3) in BGR format.
    """
    # Open PDF from bytes
    doc = fitz.open(stream=io.BytesIO(pdf_bytes).read(), filetype="pdf")
    try:
        # PyMuPDF is 0-based; caller is 1-based
        page = doc[page_num - 1]

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.h, pix.w, pix.n)

        # Drop alpha channel if present
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img
    finally:
        doc.close()


def _find_largest_quadrilateral(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the largest 4-point contour in the image.

    Returns:
        approx: np.ndarray of shape (4, 1, 2) containing 4 corner points,
                or None if no quadrilateral is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_quad = None
    best_area = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # We care only about 4-sided polygons
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best_quad = approx

    return best_quad


def _order_corners(quad: np.ndarray) -> List[List[float]]:
    """
    Order quad corners as: top-left, top-right, bottom-right, bottom-left.

    quad: np.ndarray of shape (4, 1, 2) or (4, 2)
    """
    pts = quad.reshape(4, 2).astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]     # top-left
    ordered[2] = pts[np.argmax(s)]     # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    # Convert to Python lists [[x,y], ...]
    return [[float(x), float(y)] for x, y in ordered]


def extract_pixel_coordinates(
    pdf_bytes: bytes,
    pages: list[int],
) -> Optional[Dict[int, List[List[float]]]]:
    """
    For each page in `pages`, detect the largest rectangular region
    (assumed to be the document / ID card) and return its 4 corner
    points in pixel coordinates.

    Returns:
        dict[page_number] = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        or None if nothing could be detected for any page.

    Per your rule:
        - If we hit any fatal error, we log and return None.
        - If some pages fail detection but at least one succeeds,
          we return coordinates only for the successful pages.
    """
    if not pages:
        return None

    coords: Dict[int, List[List[float]]] = {}

    try:
        for page_num in pages:
            try:
                img = _pdf_page_to_image_from_bytes(pdf_bytes, page_num, zoom=2.0)
            except Exception:
                logging.exception(
                    "Failed to render page %s for pixel extraction.", page_num
                )
                continue  # skip this page, try others

            quad = _find_largest_quadrilateral(img)
            if quad is None:
                logging.info(
                    "No rectangular document-like region detected on page %s.",
                    page_num,
                )
                continue  # nothing for this page

            ordered_corners = _order_corners(quad)
            coords[page_num] = ordered_corners

        if not coords:
            # No page produced usable coordinates
            return None

        return coords

    except Exception:
        logging.exception("Unexpected error during pixel coordinate extraction.")
        return None



=--------------------


try:
        documents = envelope_dict.get("documents_json_data", {})

        for doc_id, doc in documents.items():
            doc_type = (doc.get("doc_type") or "").lower()

            if doc_type == "paystub":
                logging.info("Running paystub adaptor for %s", doc_id)
                try:
                    # Currently we pass the whole PDF. Later we can restrict to doc["pages"].
                    extraction = process_paystub(pdf_BYTES)
                    status = extraction.get("status", "success")
                    extracted_fields = extraction.get("extracted_fields", {})
                except Exception as ex:
                    logging.exception(
                        "Error running paystub adaptor for %s: %s", doc_id, ex
                    )
                    status = "error"
                    extracted_fields = {}

                # Write into result_json for this document
                doc["result_json"] = {
                    "status": status,
                    "extracted_fields": extracted_fields,
                }
            else:
                # For all non-paystub docs, ensure there is at least a default result_json
                if not doc.get("result_json"):
                    doc["result_json"] = {
                        "status": "not_processed",
                        "extracted_fields": {},
                    }

        logging.info(
            "Envelope after employment enrichment: %s",
            json.dumps(envelope_dict),
        )
    except Exception as e:
        # Do not kill the function if enrichment fails; classification is still useful.
        logging.exception(
            "Error while updating envelope with employment data: %s", e
        )




------------------------------



import os
from io import BytesIO
from pathlib import Path
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

PDF_PATH = "./sample_paystub.pdf"
PAGES = "3"        # try "3", "1-2", "1,3-4", etc.
MODEL_ID = "prebuilt-payStub.us"

def main():
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    if not endpoint or not key:
        print("Missing AZURE_DI_ENDPOINT / AZURE_DI_KEY")
        return

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    pdf_path = Path(PDF_PATH)
    with pdf_path.open("rb") as f:
        pdf_bytes = f.read()

    poller = client.begin_analyze_document(
        model_id=MODEL_ID,
        body=BytesIO(pdf_bytes),
        pages=PAGES,                       # <-- THIS is what we're testing
        content_type="application/octet-stream",
    )
    result = poller.result()
    print("✅ DI call succeeded")
    print("num documents:", len(result.documents))

if __name__ == "__main__":
    main()


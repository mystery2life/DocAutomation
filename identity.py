# Identity_Verification_Module/__init__.py

from .api import process_identity_document

__all__ = ["process_identity_document"]


-------------Identity_Verification_Module/config.py--------

# Identity_Verification_Module/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ---- Azure DI config ----

AZURE_DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT", "").strip()
AZURE_DI_KEY = os.getenv("AZURE_DI_KEY", "").strip()

# Use prebuilt ID model by default (can override with env if needed)
ID_MODEL_ID = os.getenv("ID_MODEL_ID", "prebuilt-idDocument")

if not AZURE_DI_ENDPOINT or not AZURE_DI_KEY:
    # Fail fast at import time in Functions if env is not configured
    raise RuntimeError("AZURE_DI_ENDPOINT or AZURE_DI_KEY is not set for Identity_Verification_Module")

# ---- Field templates for each identity doc type ----
# These are the canonical field names your downstream expects.

PASSPORT_FIELDS = [
    "DocumentNumber",
    "FirstName",
    "MiddleName",
    "LastName",
    "Aliases",
    "Aliases.*",
    "DateOfBirth",
    "DateOfExpiration",
    "DateOfIssue",
    "Sex",
    "CountryRegion",
    "DocumentType",
    "Nationality",
    "PlaceOfBirth",
    "PlaceOfIssue",
    "IssuingAuthority",
    "PersonalNumber",
    "MachineReadableZone.FirstName",
    "MachineReadableZone.LastName",
    "MachineReadableZone.DocumentNumber",
    "MachineReadableZone.CountryRegion",
    "MachineReadableZone.Nationality",
    "MachineReadableZone.DateOfBirth",
    "MachineReadableZone.DateOfExpiration",
    "MachineReadableZone.Sex",
]

DL_FIELDS = [
    "CountryRegion",
    "Region",
    "DocumentNumber",
    "DocumentDiscriminator",
    "FirstName",
    "LastName",
    "Address",
    "DateOfBirth",
    "DateOfExpiration",
    "DateOfIssue",
    "EyeColor",
    "HairColor",
    "Height",
    "Weight",
    "Sex",
    "Endorsements",
    "Restrictions",
    "PersonalNumber",
    "PlaceOfBirth",
    "VehicleClassifications",
]

SSN_FIELDS = [
    "DocumentNumber",
    "FirstName",
    "LastName",
    "DateOfIssue",
]

# For now we assume Azure DI field names match canonical names.
# If your custom model uses different keys, update these maps only.

PASSPORT_FIELD_MAP = {name: name for name in PASSPORT_FIELDS}
DL_FIELD_MAP = {name: name for name in DL_FIELDS}
SSN_FIELD_MAP = {name: name for name in SSN_FIELDS}

TEMPLATE_FIELDS = {
    "passport": PASSPORT_FIELDS,
    "dl": DL_FIELDS,
    "ssn": SSN_FIELDS,
}

FIELD_MAPS = {
    "passport": PASSPORT_FIELD_MAP,
    "dl": DL_FIELD_MAP,
    "ssn": SSN_FIELD_MAP,
}

SUPPORTED_DOC_TYPES = set(TEMPLATE_FIELDS.keys())




--------------------Identity_Verification_Module/di_client.py---------------------




# Identity_Verification_Module/di_client.py

from io import BytesIO

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

from .config import AZURE_DI_ENDPOINT, AZURE_DI_KEY


# Single shared client for this module
di_client = DocumentIntelligenceClient(
    endpoint=AZURE_DI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DI_KEY),
)


def begin_analyze(model_id: str, file_bytes: bytes, **kwargs):
    """
    Wrapper to start a DI analyze operation that works with both
    document= and body= depending on SDK version.
    """
    try:
        return di_client.begin_analyze_document(
            model_id=model_id,
            document=BytesIO(file_bytes),
            **kwargs,
        )
    except TypeError:
        # Older/alternate signature
        return di_client.begin_analyze_document(
            model_id=model_id,
            body=BytesIO(file_bytes),
            **kwargs,
        )




----------------adaptor-------------


# Identity_Verification_Module/id_adapter.py

from __future__ import annotations

import logging
from typing import Dict, Any

from .config import ID_MODEL_ID, TEMPLATE_FIELDS, FIELD_MAPS, SUPPORTED_DOC_TYPES
from .di_client import begin_analyze


logger = logging.getLogger(__name__)


def _normalize_field_value(di_field) -> Dict[str, Any]:
    """
    Convert a DI DocumentField into {value, confidence}.

    - Dates are formatted as YYYY-MM-DD when value_date is present
    - Otherwise we use value if available, falling back to content.
    - Confidence returned as 0-100 float (rounded to 1 decimal).
    """
    if di_field is None:
        return {"value": None, "confidence": None}

    # Prefer typed value_date for dates
    value = None
    if getattr(di_field, "value_date", None) is not None:
        try:
            value = di_field.value_date.isoformat()
        except Exception:
            value = getattr(di_field, "content", None)
    elif getattr(di_field, "value", None) is not None:
        value = di_field.value
    else:
        value = getattr(di_field, "content", None)

    conf = getattr(di_field, "confidence", None)
    if conf is not None:
        try:
            conf = round(float(conf) * 100.0, 1)
        except Exception:
            conf = None

    return {"value": value, "confidence": conf}


def extract_identity_fields(file_bytes: bytes, doc_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Call Azure Document Intelligence ID model and return
    normalized fields for the given doc_type.

    Returns:
        dict: { canonical_field_name: { "value": ..., "confidence": ... }, ... }
    """
    doc_type_key = doc_type.lower()
    if doc_type_key not in SUPPORTED_DOC_TYPES:
        logger.info("Doc type %s not supported by Identity module", doc_type)
        return {}

    template_fields = TEMPLATE_FIELDS[doc_type_key]
    field_map = FIELD_MAPS[doc_type_key]

    poller = begin_analyze(
        model_id=ID_MODEL_ID,
        file_bytes=file_bytes,
        content_type="application/octet-stream",
        # Features are not strictly required for prebuilt-idDocument; adjust if needed
    )
    result = poller.result()

    if not getattr(result, "documents", None):
        logger.warning("No documents returned from DI for identity doc")
        return {}

    di_doc = result.documents[0]
    di_fields = getattr(di_doc, "fields", {}) or {}

    out: Dict[str, Dict[str, Any]] = {}

    for canonical_name in template_fields:
        di_name = field_map.get(canonical_name)
        di_field = di_fields.get(di_name) if di_name else None
        out[canonical_name] = _normalize_field_value(di_field)

    return out





-----------------api.py-----------


# Identity_Verification_Module/api.py

from __future__ import annotations

import logging
from typing import Dict, Any

from .config import TEMPLATE_FIELDS, SUPPORTED_DOC_TYPES
from .id_adapter import extract_identity_fields

logger = logging.getLogger(__name__)


def process_identity_document(file_bytes: bytes, doc_type: str, filename: str | None = None) -> Dict[str, Any]:
    """
    High-level API for the Identity_Verification_Module.

    Input:
        file_bytes: raw PDF bytes (already split to per-document if needed)
        doc_type:   "passport", "dl", or "ssn"
        filename:   optional, only for logging/debug

    Output shape matches what your classification envelope expects
    for result_json:

    {
      "status": "processed" | "not_processed",
      "doc_type": "<same as input>",
      "extracted_fields": {
          "<FieldName>": { "value": ..., "confidence": ... },
          ...
      }
    }
    """
    doc_type_key = (doc_type or "").lower()

    if doc_type_key not in SUPPORTED_DOC_TYPES:
        logger.info("Identity module skipping unsupported doc_type=%s", doc_type)
        return {
            "status": "not_processed",
            "doc_type": doc_type,
            "extracted_fields": {},
        }

    try:
        extracted = extract_identity_fields(file_bytes=file_bytes, doc_type=doc_type_key)
    except Exception as e:
        logger.exception("Error extracting identity fields for %s (%s)", doc_type, filename or "")
        # On error, mark as not_processed but keep template keys if you prefer.
        return {
            "status": "not_processed",
            "doc_type": doc_type,
            "extracted_fields": {},
        }

    # Build full template: all expected fields present, missing ones as None
    template_fields = TEMPLATE_FIELDS[doc_type_key]
    filled: Dict[str, Any] = {}
    for name in template_fields:
        if name in extracted:
            filled[name] = extracted[name]
        else:
            filled[name] = {"value": None, "confidence": None}

    status = "processed" if extracted else "not_processed"

    return {
        "status": status,
        "doc_type": doc_type,
        "extracted_fields": filled,
    }

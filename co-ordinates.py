"""
Configuration & canonical field templates for Employment module.
"""

# 20 Paystub fields from your paystubs sheet
PAYSTUB_FIELDS = [
    "EmployeeAddress",
    "EmployeeName",
    "EmployeeSSN",
    "EmployerAddress",
    "EmployerName",
    "PayDate",
    "PayPeriodStartDate",
    "PayPeriodEndDate",
    "CurrentPeriodGrossPay",
    "YearToDateGrossPay",
    "CurrentPeriodTaxes",
    "YearToDateTaxes",
    "CurrentPeriodDeductions",
    "YearToDateDeductions",
    "CurrentPeriodNetPay",
    "YearToDateNetPay",
    "TotalHours",
    "AveragePayRate",
    "JobTitle",
    "PayFrequency",
]

# 10 EV form fields from your EV template
EV_FIELDS = [
    "EmployeeName",
    "SSN",
    "HireDate",
    "JobTitle",
    "EIN",
    "FirstPayCheckDate",
    "CompanyName",
    "CompanyAddress",
    "AverageWorkingHours",
    "EmploymentEndDate",
]




"""
Routing helpers for Employment module.
"""


def is_paystub(doc_type: str) -> bool:
    """
    Decide if this doc_type is a paystub.
    """
    dt = (doc_type or "").lower()
    return dt in {"paystub", "payslip", "pay_stub"}


def is_ev_form(doc_type: str) -> bool:
    """
    Decide if this doc_type is an employment verification form.
    """
    dt = (doc_type or "").lower()
    return dt in {"ev_form", "employment_verification", "employmentverification"}


def is_employment_doc(doc_type: str) -> bool:
    """
    Any doc handled by this module.
    """
    return is_paystub(doc_type) or is_ev_form(doc_type)


-----result_builder----

from typing import Any, Dict, List


def _empty_template(fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Build {field: {"value": None, "confidence": None}} for all field names.
    """
    return {name: {"value": None, "confidence": None} for name in fields}


def build_processed_result_from_fields(
    template_fields: List[str],
    extracted_fields: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Map normalized fields into a standard template:

    {
      "status": "processed",
      "fields": {
         "FieldName": {"value": <>, "confidence": <>},
         ...
      }
    }

    Any field not present in extracted_fields remains value=None, confidence=None.
    """
    result_fields = _empty_template(template_fields)

    for key, item in (extracted_fields or {}).items():
        if key not in result_fields:
            # Ignore unknown fields; they can be added to template later
            continue

        if isinstance(item, dict):
            value = item.get("value")
            confidence = item.get("confidence")
        else:
            value = item
            confidence = None

        result_fields[key]["value"] = value
        result_fields[key]["confidence"] = confidence

    return {
        "status": "processed",
        "fields": result_fields,
    }


def build_not_processed_result() -> Dict[str, Any]:
    """
    Used for doc types this module does not process yet.
    """
    return {
        "status": "not_processed",
        "fields": {},
    }


def build_error_result(error_msg: str) -> Dict[str, Any]:
    """
    Used when a runtime error occurs inside employment processing.
    """
    return {
        "status": "error",
        "fields": {},
        "error_message": str(error_msg),
    }


-----di_client-----

import os
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


@lru_cache(maxsize=1)
def get_di_client() -> DocumentIntelligenceClient:
    """
    Returns a cached Document Intelligence client, using DI_ENDPOINT and DI_KEY env vars.
    """
    endpoint = os.getenv("DI_ENDPOINT")
    key = os.getenv("DI_KEY")

    if not endpoint or not key:
        raise RuntimeError("DI_ENDPOINT or DI_KEY environment variables are not set.")

    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))




------employment_module/paystub/adaptor.py-----



  import os
import logging
from typing import Any, Dict

from employment_module.di_client import get_di_client
from employment_module.config import PAYSTUB_FIELDS


def _field_to_dict(field) -> Dict[str, Any]:
    """
    Convert a DI field to our {value, confidence} dict.
    """
    if field is None:
        return {"value": None, "confidence": None}

    value = getattr(field, "value", None)
    if value is None:
        value = getattr(field, "content", None)

    return {
        "value": value,
        "confidence": getattr(field, "confidence", None),
    }


def extract_paystub_structured(pdf_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """
    Calls Azure Document Intelligence payStub prebuilt model and returns
    a mapping of PAYSTUB_FIELDS -> {value, confidence}.
    """
    client = get_di_client()
    model_id = os.getenv("PAYSTUB_MODEL_ID", "prebuilt-payStub.us")

    logging.info("Submitting payStub analysis to Document Intelligence (model_id=%s).", model_id)

    poller = client.begin_analyze_document(
        model_id=model_id,
        document=pdf_bytes,
    )
    result = poller.result()

    if not result.documents:
        logging.warning("payStub model returned no documents.")
        return {}

    doc = result.documents[0]
    fields = doc.fields or {}

    out: Dict[str, Dict[str, Any]] = {}

    for name in PAYSTUB_FIELDS:
        di_field = fields.get(name)
        out[name] = _field_to_dict(di_field)

    logging.info("Extracted payStub structured fields: %s", out)
    return out


------paystub normalize------

import re
from datetime import datetime
from typing import Any, Dict

from employment_module.config import PAYSTUB_FIELDS


def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _parse_date(val: Any) -> str | None:
    """
    Try to parse various date formats and return ISO 'YYYY-MM-DD'.
    If parsing fails, return None.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # If nothing works, just return the original string
    return s


def _parse_money(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val)
    s = s.replace("$", "").replace(",", "").strip()
    s = s.replace(" ", "")
    if not s:
        return None
    # Handle parentheses for negative values: (123.45)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _title_case_name(val: Any) -> str | None:
    s = _clean_str(val)
    if s is None:
        return None
    return " ".join(part.capitalize() for part in s.split())


def _normalize_address(val: Any) -> str | None:
    # For now, just clean whitespace
    return _clean_str(val)


def _normalize_ssn(val: Any) -> str | None:
    s = _clean_str(val)
    if s is None:
        return None
    # Keep only digits or X and mask if needed
    s = s.replace("-", "").replace(" ", "")
    if len(s) == 9:
        return f"{s[0:3]}-{s[3:5]}-{s[5:9]}"
    return s


def normalize_paystub_fields(
    extracted: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize all paystub fields into clean values.

    Input:  {"FieldName": {"value": raw, "confidence": x}, ...}
    Output: {"FieldName": {"value": cleaned, "confidence": x}, ...}
    """
    out: Dict[str, Dict[str, Any]] = {}

    for field_name in PAYSTUB_FIELDS:
        item = (extracted or {}).get(field_name, {}) or {}
        raw_val = item.get("value")
        conf = item.get("confidence")

        cleaned_val: Any = raw_val

        if field_name in {"EmployeeName", "EmployerName", "JobTitle"}:
            cleaned_val = _title_case_name(raw_val)
        elif field_name in {"EmployeeAddress", "EmployerAddress"}:
            cleaned_val = _normalize_address(raw_val)
        elif field_name == "EmployeeSSN":
            cleaned_val = _normalize_ssn(raw_val)
        elif field_name in {"PayDate", "PayPeriodStartDate", "PayPeriodEndDate"}:
            cleaned_val = _parse_date(raw_val)
        elif field_name in {
            "CurrentPeriodGrossPay",
            "YearToDateGrossPay",
            "CurrentPeriodTaxes",
            "YearToDateTaxes",
            "CurrentPeriodDeductions",
            "YearToDateDeductions",
            "CurrentPeriodNetPay",
            "YearToDateNetPay",
        }:
            cleaned_val = _parse_money(raw_val)
        elif field_name in {"TotalHours", "AveragePayRate"}:
            cleaned_val = _parse_float(raw_val)
        elif field_name == "PayFrequency":
            # Try to normalize to one of: Weekly, Biweekly, Semimonthly, Monthly
            s = _clean_str(raw_val)
            if s:
                lower = s.lower()
                if "week" in lower:
                    if "bi" in lower or "2" in lower:
                        cleaned_val = "Biweekly"
                    else:
                        cleaned_val = "Weekly"
                elif "semi" in lower:
                    cleaned_val = "Semimonthly"
                elif "month" in lower:
                    cleaned_val = "Monthly"
                else:
                    cleaned_val = s
            else:
                cleaned_val = None
        else:
            cleaned_val = _clean_str(raw_val)

        out[field_name] = {
            "value": cleaned_val,
            "confidence": conf,
        }

    return out


-------------employment_module/paystub/processor.py------------------------------



from typing import Any, Dict

from employment_module.config import PAYSTUB_FIELDS
from employment_module.result_builder import build_processed_result_from_fields
from employment_module.paystub.adaptor import extract_paystub_structured
from employment_module.paystub.normalize import normalize_paystub_fields


def process_paystub_document(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    High-level wrapper for a single paystub document.

    1. Call paystub adaptor (Document Intelligence prebuilt-payStub)
    2. Normalize all extracted fields
    3. Wrap into standard result_json
    """
    extracted = extract_paystub_structured(pdf_bytes) or {}
    normalized = normalize_paystub_fields(extracted)
    return build_processed_result_from_fields(
        template_fields=PAYSTUB_FIELDS,
        extracted_fields=normalized,
    )


----------------ev_form/adaptor.py--------------------------------------

import os
import logging
from typing import Any, Dict

from employment_module.di_client import get_di_client
from employment_module.config import EV_FIELDS


def _field_to_dict(field) -> Dict[str, Any]:
    """
    Convert a DI field to our {value, confidence} dict.
    """
    if field is None:
        return {"value": None, "confidence": None}

    value = getattr(field, "value", None)
    if value is None:
        value = getattr(field, "content", None)

    return {
        "value": value,
        "confidence": getattr(field, "confidence", None),
    }


def extract_ev_structured(pdf_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """
    Calls custom EV model in Document Intelligence and returns a mapping of
    EV_FIELDS -> {value, confidence}.
    """
    client = get_di_client()
    model_id = os.getenv("EV_MODEL_ID")

    if not model_id:
        raise RuntimeError("EV_MODEL_ID environment variable is not set.")

    logging.info("Submitting EV analysis to Document Intelligence model '%s'.", model_id)
    poller = client.begin_analyze_document(
        model_id=model_id,
        document=pdf_bytes,
    )
    result = poller.result()

    if not result.documents:
        logging.warning("EV model returned no documents.")
        return {}

    doc = result.documents[0]
    fields = doc.fields or {}

    out: Dict[str, Dict[str, Any]] = {}

    for name in EV_FIELDS:
        di_field = fields.get(name)
        out[name] = _field_to_dict(di_field)

    logging.info("Extracted EV structured fields: %s", out)
    return out




------------------ev_form/normalize.py --------------------------

import re
from datetime import datetime
from typing import Any, Dict

from employment_module.config import EV_FIELDS


def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _parse_date(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def _normalize_ssn(val: Any) -> str | None:
    s = _clean_str(val)
    if s is None:
        return None
    s = s.replace("-", "").replace(" ", "")
    if len(s) == 9:
        return f"{s[0:3]}-{s[3:5]}-{s[5:9]}"
    return s


def _normalize_ein(val: Any) -> str | None:
    s = _clean_str(val)
    if s is None:
        return None
    s = s.replace("-", "").replace(" ", "")
    if len(s) == 9:
        return f"{s[0:2]}-{s[2:9]}"
    return s


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _title_case(val: Any) -> str | None:
    s = _clean_str(val)
    if s is None:
        return None
    return " ".join(part.capitalize() for part in s.split())


def _normalize_address(val: Any) -> str | None:
    return _clean_str(val)


def normalize_ev_fields(
    extracted: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize all EV form fields into clean values.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for field_name in EV_FIELDS:
        item = (extracted or {}).get(field_name, {}) or {}
        raw_val = item.get("value")
        conf = item.get("confidence")

        cleaned_val: Any = raw_val

        if field_name in {"EmployeeName", "JobTitle", "CompanyName"}:
            cleaned_val = _title_case(raw_val)
        elif field_name == "CompanyAddress":
            cleaned_val = _normalize_address(raw_val)
        elif field_name == "SSN":
            cleaned_val = _normalize_ssn(raw_val)
        elif field_name == "EIN":
            cleaned_val = _normalize_ein(raw_val)
        elif field_name in {"HireDate", "FirstPayCheckDate", "EmploymentEndDate"}:
            cleaned_val = _parse_date(raw_val)
        elif field_name == "AverageWorkingHours":
            cleaned_val = _parse_float(raw_val)
        else:
            cleaned_val = _clean_str(raw_val)

        out[field_name] = {
            "value": cleaned_val,
            "confidence": conf,
        }

    return out



---------------------ev_form/processor.py ------------------------- 


from typing import Any, Dict

from employment_module.config import EV_FIELDS
from employment_module.result_builder import build_processed_result_from_fields
from employment_module.ev_form.adaptor import extract_ev_structured
from employment_module.ev_form.normalize import normalize_ev_fields


def process_ev_document(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    High-level wrapper for a single EV form document.

    1. Call EV adaptor (custom DI model)
    2. Normalize all extracted fields
    3. Wrap into standard result_json
    """
    extracted = extract_ev_structured(pdf_bytes) or {}
    normalized = normalize_ev_fields(extracted)
    return build_processed_result_from_fields(
        template_fields=EV_FIELDS,
        extracted_fields=normalized,
    )


-------------------------employmentmodueinit-----------------



import logging
from typing import Dict, Any

from employment_module.routing import is_employment_doc, is_paystub, is_ev_form
from employment_module.result_builder import (
    build_not_processed_result,
    build_error_result,
)
from employment_module.paystub.processor import process_paystub_document
from employment_module.ev_form.processor import process_ev_document


def process_envelope(envelope_dict: dict, pdf_bytes: bytes) -> dict:
    """
    For each document in the classification envelope that is an employment
    doc (paystub / EV form), fill doc['result_json'].

    Currently, we run each adaptor on the full PDF bytes. If later you want
    per-document page cropping, it can be added here.
    """

    documents = envelope_dict.get("documents", {}) or {}
    blob_path = envelope_dict.get("blob_path", "")
    logging.info("Starting employment processing for blob_path='%s'", blob_path)

    for doc_id, doc in documents.items():
        doc_type = (doc.get("doc_type") or "").lower()

        if not is_employment_doc(doc_type):
            continue

        logging.info("Processing employment document '%s' of type '%s'", doc_id, doc_type)

        try:
            if is_paystub(doc_type):
                result_json = process_paystub_document(pdf_bytes)
            elif is_ev_form(doc_type):
                result_json = process_ev_document(pdf_bytes)
            else:
                result_json = build_not_processed_result()
        except Exception as e:
            logging.exception("Error while processing employment doc '%s': %s", doc_id, e)
            result_json = build_error_result(str(e))

        doc["result_json"] = result_json
        documents[doc_id] = doc

    envelope_dict["documents"] = documents
    logging.info("Completed employment processing.")
    return envelope_dict


------------------------fucntion app----------------------




# -------------------------------------------------------
    # 4) Fill result_json for employment docs (paystub + EV)
    # -------------------------------------------------------
    try:
        envelope_dict = employment_process_envelope(
            envelope_dict=envelope_dict,
            pdf_bytes=pdf_bytes,
        )
    except Exception as e:
        logging.exception("Error in employment_module.process_envelope: %s", e)
        raise

    # -------------------------------------------------------
    # 5) Log final envelope JSON (and later send downstream)
    # -------------------------------------------------------
    try:
        final_json = json.dumps(envelope_dict)
        logging.info("Final enriched classification envelope: %s", final_json)

        # Here you can write final_json to another queue, blob, or DB if needed.
    except Exception as e:
        logging.exception("Error while serializing/logging final envelope: %s", e)
        raise



-------------------------------json-----------------


{
  "job_uuid": "2bc44039-59ba-4392-8e51-9e113aa0061e",
  "processed_utc": "2025-12-09T17:41:32Z",
  "blob_path": "ingest/2025/12/09/2019YTD-payroll.pdf",
  "total_pages": 20,
  "documents_data": [
    {
      "doc_id": "DOC_001",
      "doc_type": "paystub",
      "pages": [1, 2],
      "doc_pixel_coordinates": {
        "1": [[100, 100], [500, 100], [500, 400], [100, 400]],
        "2": [[110, 120], [510, 120], [510, 420], [110, 420]]
      },
      "result_json": {
        "status": "success",
        "extracted_fields": {
          "EmployeeAddress": { "value": "1234 Concord St, Nashua, NH 03064", "confidence": 42.4 },
          "EmployeeName": { "value": "John Doe", "confidence": 96.1 },
          "EmployeeSSN": { "value": "XXX-XX-1234", "confidence": 92.0 },
          "EmployerAddress": { "value": "99 Main St, Anytown, NH 10000", "confidence": 70.5 },
          "EmployerName": { "value": "Google Inc.", "confidence": 85.0 },
          "PayDate": { "value": "2025-10-13", "confidence": 77.3 },
          "PayPeriodStartDate": { "value": "2025-10-06", "confidence": 62.9 },
          "PayPeriodEndDate": { "value": "2025-10-13", "confidence": 93.3 },
          "CurrentPeriodGrossPay": { "value": "828.00", "confidence": 97.6 },
          "YearToDateGrossPay": { "value": "23280.00", "confidence": 92.2 },
          "CurrentPeriodTaxes": { "value": null, "confidence": null },
          "YearToDateTaxes": { "value": null, "confidence": null },
          "CurrentPeriodDeductions": { "value": "414.43", "confidence": 95.3 },
          "YearToDateDeductions": { "value": "6630.88", "confidence": 90.5 },
          "CurrentPeriodNetPay": { "value": "413.57", "confidence": 97.2 },
          "YearToDateNetPay": { "value": "16649.12", "confidence": 90.0 },
          "TotalHours": { "value": 69, "confidence": 100.0 },
          "AveragePayRate": { "value": 12.0, "confidence": 100.0 },
          "JobTitle": { "value": null, "confidence": null },
          "PayFrequency": { "value": null, "confidence": null }
        }
      }
    },
    {
      "doc_id": "DOC_002",
      "doc_type": "bank_stmt",
      "pages": [3, 10],
      "doc_pixel_coordinates": {
        "3": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "4": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "5": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "6": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "7": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "8": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "9": [[90, 80], [550, 80], [550, 700], [90, 700]],
        "10": [[90, 80], [550, 80], [550, 700], [90, 700]]
      },
      "result_json": {
        "status": "not_processed",
        "extracted_fields": {}
      }
    },
    {
      "doc_id": "DOC_003",
      "doc_type": "passport",
      "pages": [11],
      "doc_pixel_coordinates": {
        "11": [[120, 90], [520, 90], [520, 380], [120, 380]]
      },
      "result_json": {
        "status": "success",
        "extracted_fields": {
          "DocumentNumber": { "value": "X1234567", "confidence": 98.0 },
          "FirstName": { "value": "JOHN", "confidence": 98.0 },
          "MiddleName": { "value": "A", "confidence": 95.0 },
          "LastName": { "value": "DOE", "confidence": 98.0 },
          "Aliases": { "value": null, "confidence": null },
          "Aliases.*": { "value": null, "confidence": null },
          "DateOfBirth": { "value": "1990-01-15", "confidence": 97.0 },
          "DateOfExpiration": { "value": "2030-01-14", "confidence": 97.0 },
          "DateOfIssue": { "value": "2020-01-15", "confidence": 97.0 },
          "Sex": { "value": "M", "confidence": 99.0 },
          "CountryRegion": { "value": "USA", "confidence": 99.0 },
          "DocumentType": { "value": "P", "confidence": 99.0 },
          "Nationality": { "value": "USA", "confidence": 99.0 },
          "PlaceOfBirth": { "value": "NASHUA", "confidence": 96.0 },
          "PlaceOfIssue": { "value": "BOSTON", "confidence": 96.0 },
          "IssuingAuthority": { "value": "US DEPARTMENT OF STATE", "confidence": 96.0 },
          "PersonalNumber": { "value": null, "confidence": null },
          "MachineReadableZone.FirstName": { "value": "JOHN", "confidence": 98.0 },
          "MachineReadableZone.LastName": { "value": "DOE", "confidence": 98.0 },
          "MachineReadableZone.DocumentNumber": { "value": "X1234567", "confidence": 98.0 },
          "MachineReadableZone.CountryRegion": { "value": "USA", "confidence": 98.0 },
          "MachineReadableZone.Nationality": { "value": "USA", "confidence": 98.0 },
          "MachineReadableZone.DateOfBirth": { "value": "1990-01-15", "confidence": 98.0 },
          "MachineReadableZone.DateOfExpiration": { "value": "2030-01-14", "confidence": 98.0 },
          "MachineReadableZone.Sex": { "value": "M", "confidence": 98.0 }
        }
      }
    },
    {
      "doc_id": "DOC_004",
      "doc_type": "dl",
      "pages": [12],
      "doc_pixel_coordinates": {
        "12": [[130, 100], [530, 100], [530, 360], [130, 360]]
      },
      "result_json": {
        "status": "success",
        "extracted_fields": {
          "CountryRegion": { "value": "USA", "confidence": 99.0 },
          "Region": { "value": "NH", "confidence": 99.0 },
          "DocumentNumber": { "value": "D123456789", "confidence": 98.0 },
          "DocumentDiscriminator": { "value": "A1B2C3", "confidence": 90.0 },
          "FirstName": { "value": "JOHN", "confidence": 98.0 },
          "LastName": { "value": "DOE", "confidence": 98.0 },
          "Address": { "value": "1234 Concord St, Nashua, NH 03064", "confidence": 95.0 },
          "DateOfBirth": { "value": "1990-01-15", "confidence": 97.0 },
          "DateOfExpiration": { "value": "2031-10-01", "confidence": 97.0 },
          "DateOfIssue": { "value": "2023-10-01", "confidence": 97.0 },
          "EyeColor": { "value": "BRN", "confidence": 95.0 },
          "HairColor": { "value": "BLK", "confidence": 95.0 },
          "Height": { "value": "5'11\"", "confidence": 95.0 },
          "Weight": { "value": "180", "confidence": 95.0 },
          "Sex": { "value": "M", "confidence": 99.0 },
          "Endorsements": { "value": null, "confidence": null },
          "Restrictions": { "value": null, "confidence": null },
          "PersonalNumber": { "value": null, "confidence": null },
          "PlaceOfBirth": { "value": "NASHUA", "confidence": 90.0 },
          "VehicleClassifications": { "value": "D", "confidence": 90.0 }
        }
      }
    },
    {
      "doc_id": "DOC_005",
      "doc_type": "ssn",
      "pages": [13],
      "doc_pixel_coordinates": {
        "13": [[140, 110], [540, 110], [540, 260], [140, 260]]
      },
      "result_json": {
        "status": "success",
        "extracted_fields": {
          "DocumentNumber": { "value": "XXX-XX-4321", "confidence": 98.0 },
          "FirstName": { "value": "JOHN", "confidence": 98.0 },
          "LastName": { "value": "DOE", "confidence": 98.0 },
          "DateOfIssue": { "value": "2015-05-06", "confidence": 91.6 }
        }
      }
    }
  ]
}



-------------------------



{
  "job_uuid": "2bc4403-59ba-4392-8e51-9e113aa0601e",
  "processed_utc": "2025-12-09T07:13:28Z",
  "blob_path": "ingest/2025/12/09/201901fb-0e76-4740-8d82-sample.pdf",
  "total_pages": 17,
  "documents_data": [
    {
      "doc_id": "DOC_ID1",
      "doc_type": "paystub",
      "pages": [1, 2],
      "doc_pixel_coordinates": {
        "1": [[110, 90], [1910, 90], [1910, 1080], [110, 1080]],
        "2": [[120, 95], [1900, 95], [1900, 1075], [120, 1075]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "paystub",
        "extracted_fields": {
          "EmployeeAddress":   { "value": "1234 Concord St, Nashua, NH 03064", "confidence": 84.2 },
          "EmployeeName":      { "value": "John Doe",                         "confidence": 96.1 },
          "EmployeeSSN":       { "value": "700-XX-1234",                      "confidence": 92.0 },
          "EmployerAddress":   { "value": "99 Main St, Anytown, NH 10000",    "confidence": 70.5 },
          "EmployerName":      { "value": "Google Inc",                       "confidence": 85.0 },
          "PayDate":           { "value": "2025-10-13",                       "confidence": 77.3 },
          "PayPeriodStartDate":{ "value": "2025-10-06",                       "confidence": 62.9 },
          "PayPeriodEndDate":  { "value": "2025-10-13",                       "confidence": 93.3 },
          "CurrentPeriodGrossPay": { "value": "828.07",                       "confidence": 97.6 },
          "YearToDateGrossPay":    { "value": "23280.00",                     "confidence": 92.2 },
          "CurrentPeriodTaxes":    { "value": "210.50",                       "confidence": 90.1 },
          "YearToDateTaxes":       { "value": "4430.12",                      "confidence": 91.8 },
          "CurrentPeriodDeductions": { "value": "414.43",                     "confidence": 95.3 },
          "YearToDateDeductions":    { "value": "6630.88",                    "confidence": 90.5 },
          "CurrentPeriodNetPay":     { "value": "413.57",                     "confidence": 97.2 },
          "YearToDateNetPay":        { "value": "16649.12",                   "confidence": 90.0 },
          "TotalHours":              { "value": 69,                           "confidence": 100.0 },
          "AveragePayRate":          { "value": 12.0,                         "confidence": 100.0 },
          "JobTitle":                { "value": "Software Engineer",          "confidence": 88.7 },
          "PayFrequency":            { "value": "BIWEEKLY",                   "confidence": 93.9 }
        }
      }
    },
    {
      "doc_id": "DOC_ID2",
      "doc_type": "bank_statement",
      "pages": [3, 4, 5, 6],
      "doc_pixel_coordinates": {
        "3": [[100, 80], [1910, 80], [1910, 1070], [100, 1070]],
        "4": [[105, 82], [1905, 82], [1905, 1072], [105, 1072]],
        "5": [[110, 84], [1900, 84], [1900, 1074], [110, 1074]],
        "6": [[115, 86], [1895, 86], [1895, 1076], [115, 1076]]
      },
      "result_json": {
        "status": "not_processed",
        "doc_type": "bank_statement",
        "extracted_fields": {}
      }
    },
    {
      "doc_id": "DOC_ID3",
      "doc_type": "paystub",
      "pages": [7, 8],
      "doc_pixel_coordinates": {
        "7": [[120, 100], [1910, 100], [1910, 1070], [120, 1070]],
        "8": [[130, 105], [1900, 105], [1900, 1065], [130, 1065]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "paystub",
        "extracted_fields": {
          "EmployeeAddress":   { "value": "987 Elm St, Manchester, NH 03101", "confidence": 82.6 },
          "EmployeeName":      { "value": "Jane Smith",                       "confidence": 95.4 },
          "EmployeeSSN":       { "value": "701-XX-5678",                      "confidence": 91.2 },
          "EmployerAddress":   { "value": "500 Tech Park, Boston, MA 02110",  "confidence": 73.1 },
          "EmployerName":      { "value": "Acme Corp",                        "confidence": 87.5 },
          "PayDate":           { "value": "2025-09-30",                       "confidence": 79.4 },
          "PayPeriodStartDate":{ "value": "2025-09-23",                       "confidence": 68.8 },
          "PayPeriodEndDate":  { "value": "2025-09-30",                       "confidence": 92.7 },
          "CurrentPeriodGrossPay": { "value": "1542.35",                      "confidence": 96.9 },
          "YearToDateGrossPay":    { "value": "30120.50",                     "confidence": 93.1 },
          "CurrentPeriodTaxes":    { "value": "389.72",                       "confidence": 89.5 },
          "YearToDateTaxes":       { "value": "6120.33",                      "confidence": 90.4 },
          "CurrentPeriodDeductions": { "value": "210.40",                     "confidence": 94.6 },
          "YearToDateDeductions":    { "value": "4120.99",                    "confidence": 91.3 },
          "CurrentPeriodNetPay":     { "value": "942.23",                     "confidence": 96.2 },
          "YearToDateNetPay":        { "value": "19879.18",                   "confidence": 92.4 },
          "TotalHours":              { "value": 80,                           "confidence": 99.1 },
          "AveragePayRate":          { "value": 19.3,                         "confidence": 98.7 },
          "JobTitle":                { "value": "Data Analyst",               "confidence": 89.9 },
          "PayFrequency":            { "value": "WEEKLY",                     "confidence": 92.1 }
        }
      }
    },
    {
      "doc_id": "DOC_ID4",
      "doc_type": "passport",
      "pages": [9],
      "doc_pixel_coordinates": {
        "9": [[340, 220], [1580, 220], [1580, 900], [340, 900]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "passport",
        "extracted_fields": {
          "DocumentNumber":                 { "value": "X1234567",   "confidence": 98.4 },
          "FirstName":                      { "value": "John",       "confidence": 97.1 },
          "MiddleName":                     { "value": "A",          "confidence": 88.6 },
          "LastName":                       { "value": "Doe",        "confidence": 98.0 },
          "Aliases":                        { "value": ["Johnny"],   "confidence": 72.9 },
          "Aliases.*":                      { "value": "Johnny",     "confidence": 72.9 },
          "DateOfBirth":                    { "value": "1990-05-14", "confidence": 96.3 },
          "DateOfExpiration":               { "value": "2030-05-13", "confidence": 95.7 },
          "DateOfIssue":                    { "value": "2020-05-14", "confidence": 94.1 },
          "Sex":                            { "value": "M",          "confidence": 99.0 },
          "CountryRegion":                  { "value": "USA",        "confidence": 99.2 },
          "DocumentType":                   { "value": "P",          "confidence": 98.8 },
          "Nationality":                    { "value": "USA",        "confidence": 99.2 },
          "PlaceOfBirth":                   { "value": "Boston, MA", "confidence": 90.6 },
          "PlaceOfIssue":                   { "value": "Boston, MA", "confidence": 89.7 },
          "IssuingAuthority":              { "value": "U.S. Department of State", "confidence": 93.8 },
          "PersonalNumber":                { "value": "999999999",  "confidence": 81.5 },
          "MachineReadableZone.FirstName": { "value": "JOHN",       "confidence": 97.9 },
          "MachineReadableZone.LastName":  { "value": "DOE",        "confidence": 97.9 },
          "MachineReadableZone.DocumentNumber": { "value": "X1234567", "confidence": 98.3 },
          "MachineReadableZone.CountryRegion":  { "value": "USA",    "confidence": 99.1 },
          "MachineReadableZone.Nationality":    { "value": "USA",    "confidence": 99.1 },
          "MachineReadableZone.DateOfBirth":    { "value": "900514", "confidence": 96.0 },
          "MachineReadableZone.DateOfExpiration": { "value": "300513", "confidence": 95.5 },
          "MachineReadableZone.Sex":             { "value": "M",     "confidence": 98.9 }
        }
      }
    },
    {
      "doc_id": "DOC_ID5",
      "doc_type": "dl",
      "pages": [10],
      "doc_pixel_coordinates": {
        "10": [[360, 260], [1560, 260], [1560, 860], [360, 860]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "dl",
        "extracted_fields": {
          "CountryRegion":        { "value": "USA",            "confidence": 99.0 },
          "Region":               { "value": "NH",             "confidence": 98.2 },
          "DocumentNumber":       { "value": "D123456789",     "confidence": 97.6 },
          "DocumentDiscriminator":{ "value": "01-23-45",       "confidence": 84.3 },
          "FirstName":            { "value": "John",           "confidence": 96.8 },
          "LastName":             { "value": "Doe",            "confidence": 97.3 },
          "Address":              { "value": "1234 Concord St, Nashua, NH 03064", "confidence": 92.4 },
          "DateOfBirth":          { "value": "1990-05-14",     "confidence": 96.5 },
          "DateOfExpiration":     { "value": "2029-05-14",     "confidence": 95.2 },
          "DateOfIssue":          { "value": "2019-05-14",     "confidence": 94.7 },
          "EyeColor":             { "value": "BRO",            "confidence": 93.1 },
          "HairColor":            { "value": "BLK",            "confidence": 92.0 },
          "Height":               { "value": "5'11\"",         "confidence": 91.4 },
          "Weight":               { "value": "183 lb",         "confidence": 88.9 },
          "Sex":                  { "value": "M",              "confidence": 99.2 },
          "Endorsements":         { "value": "NONE",           "confidence": 80.3 },
          "Restrictions":         { "value": "CORRECTIVE LENSES", "confidence": 86.7 },
          "PersonalNumber":       { "value": "123456789",      "confidence": 82.5 },
          "PlaceOfBirth":         { "value": "Boston, MA",     "confidence": 78.4 },
          "VehicleClassifications": { "value": "CLASS D",      "confidence": 93.0 }
        }
      }
    },
    {
      "doc_id": "DOC_ID6",
      "doc_type": "ssn",
      "pages": [11],
      "doc_pixel_coordinates": {
        "11": [[400, 280], [1520, 280], [1520, 820], [400, 820]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "ssn",
        "extracted_fields": {
          "DocumentNumber": { "value": "123-45-6789", "confidence": 97.1 },
          "FirstName":      { "value": "John",        "confidence": 95.8 },
          "LastName":       { "value": "Doe",         "confidence": 96.3 },
          "DateOfIssue":    { "value": "2010-03-15",  "confidence": 83.9 }
        }
      }
    },
    {
      "doc_id": "DOC_ID7",
      "doc_type": "unknown",
      "pages": [12],
      "doc_pixel_coordinates": {
        "12": [[150, 90], [1850, 90], [1850, 1040], [150, 1040]]
      },
      "result_json": {
        "status": "not_processed",
        "doc_type": "unknown",
        "extracted_fields": {}
      }
    },
    {
      "doc_id": "DOC_ID8",
      "doc_type": "bank_statement",
      "pages": [13, 14],
      "doc_pixel_coordinates": {
        "13": [[130, 80], [1870, 80], [1870, 1045], [130, 1045]],
        "14": [[135, 82], [1865, 82], [1865, 1047], [135, 1047]]
      },
      "result_json": {
        "status": "not_processed",
        "doc_type": "bank_statement",
        "extracted_fields": {}
      }
    },
    {
      "doc_id": "DOC_ID9",
      "doc_type": "unknown",
      "pages": [15],
      "doc_pixel_coordinates": {
        "15": [[160, 100], [1840, 100], [1840, 1030], [160, 1030]]
      },
      "result_json": {
        "status": "not_processed",
        "doc_type": "unknown",
        "extracted_fields": {}
      }
    },
    {
      "doc_id": "DOC_ID10",
      "doc_type": "passport",
      "pages": [16],
      "doc_pixel_coordinates": {
        "16": [[350, 230], [1570, 230], [1570, 910], [350, 910]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "passport",
        "extracted_fields": {
          "DocumentNumber":                 { "value": "Y7654321",   "confidence": 98.1 },
          "FirstName":                      { "value": "Jane",       "confidence": 97.4 },
          "MiddleName":                     { "value": "B",          "confidence": 87.2 },
          "LastName":                       { "value": "Smith",      "confidence": 97.9 },
          "Aliases":                        { "value": [],           "confidence": 50.0 },
          "Aliases.*":                      { "value": null,         "confidence": 0.0 },
          "DateOfBirth":                    { "value": "1992-11-02", "confidence": 96.0 },
          "DateOfExpiration":               { "value": "2032-11-01", "confidence": 95.6 },
          "DateOfIssue":                    { "value": "2022-11-02", "confidence": 94.2 },
          "Sex":                            { "value": "F",          "confidence": 99.1 },
          "CountryRegion":                  { "value": "USA",        "confidence": 99.3 },
          "DocumentType":                   { "value": "P",          "confidence": 98.6 },
          "Nationality":                    { "value": "USA",        "confidence": 99.3 },
          "PlaceOfBirth":                   { "value": "Seattle, WA","confidence": 90.3 },
          "PlaceOfIssue":                   { "value": "Seattle, WA","confidence": 89.5 },
          "IssuingAuthority":               { "value": "U.S. Department of State", "confidence": 93.5 },
          "PersonalNumber":                 { "value": "888888888",  "confidence": 80.9 },
          "MachineReadableZone.FirstName":  { "value": "JANE",       "confidence": 97.8 },
          "MachineReadableZone.LastName":   { "value": "SMITH",      "confidence": 97.8 },
          "MachineReadableZone.DocumentNumber": { "value": "Y7654321", "confidence": 98.0 },
          "MachineReadableZone.CountryRegion":  { "value": "USA",    "confidence": 99.0 },
          "MachineReadableZone.Nationality":    { "value": "USA",    "confidence": 99.0 },
          "MachineReadableZone.DateOfBirth":    { "value": "921102", "confidence": 95.9 },
          "MachineReadableZone.DateOfExpiration": { "value": "321101", "confidence": 95.4 },
          "MachineReadableZone.Sex":            { "value": "F",      "confidence": 98.8 }
        }
      }
    },
    {
      "doc_id": "DOC_ID11",
      "doc_type": "dl",
      "pages": [17],
      "doc_pixel_coordinates": {
        "17": [[365, 255], [1555, 255], [1555, 855], [365, 855]]
      },
      "result_json": {
        "status": "processed",
        "doc_type": "dl",
        "extracted_fields": {
          "CountryRegion":        { "value": "USA",            "confidence": 99.1 },
          "Region":               { "value": "MA",             "confidence": 97.9 },
          "DocumentNumber":       { "value": "S987654321",     "confidence": 97.2 },
          "DocumentDiscriminator":{ "value": "11-22-33",       "confidence": 83.6 },
          "FirstName":            { "value": "Jane",           "confidence": 96.2 },
          "LastName":             { "value": "Smith",          "confidence": 96.9 },
          "Address":              { "value": "45 Harbor Way, Boston, MA 02110", "confidence": 91.7 },
          "DateOfBirth":          { "value": "1992-11-02",     "confidence": 96.0 },
          "DateOfExpiration":     { "value": "2030-11-02",     "confidence": 95.1 },
          "DateOfIssue":          { "value": "2020-11-02",     "confidence": 94.3 },
          "EyeColor":             { "value": "BLU",            "confidence": 92.7 },
          "HairColor":            { "value": "BRN",            "confidence": 92.2 },
          "Height":               { "value": "5'6\"",          "confidence": 90.8 },
          "Weight":               { "value": "140 lb",         "confidence": 88.1 },
          "Sex":                  { "value": "F",              "confidence": 99.0 },
          "Endorsements":         { "value": "NONE",           "confidence": 79.8 },
          "Restrictions":         { "value": "NONE",           "confidence": 82.9 },
          "PersonalNumber":       { "value": "987654321",      "confidence": 81.2 },
          "PlaceOfBirth":         { "value": "Seattle, WA",    "confidence": 77.5 },
          "VehicleClassifications": { "value": "CLASS D",      "confidence": 92.8 }
        }
      }
    }
  ]
}

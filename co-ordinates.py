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

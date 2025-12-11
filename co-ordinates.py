# employment_module/config.py

# Canonical field order for paystubs.
PAYSTUB_FIELD_ORDER = [
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

# Types – used for normalization.
PAYSTUB_FIELD_TYPES = {
    "EmployeeAddress": "string",
    "EmployeeName": "string",
    "EmployeeSSN": "string",
    "EmployerAddress": "string",
    "EmployerName": "string",
    "PayDate": "date",
    "PayPeriodStartDate": "date",
    "PayPeriodEndDate": "date",
    "CurrentPeriodGrossPay": "number",
    "YearToDateGrossPay": "number",
    "CurrentPeriodTaxes": "number",
    "YearToDateTaxes": "number",
    "CurrentPeriodDeductions": "number",
    "YearToDateDeductions": "number",
    "CurrentPeriodNetPay": "number",
    "YearToDateNetPay": "number",
    "TotalHours": "number",
    "AveragePayRate": "number",
    "JobTitle": "string",
    "PayFrequency": "string",
}

# Aliases: DI / LLM field names → canonical names above.
PAYSTUB_ALIASES = {
    # DI aliases (add / tweak based on your actual DI field names):
    "EmployeeAddress": "EmployeeAddress",
    "EmployeeName": "EmployeeName",
    "EmployeeSSN": "EmployeeSSN",
    "EmployerAddress": "EmployerAddress",
    "EmployerName": "EmployerName",
    "PayDate": "PayDate",
    "PayPeriodStartDate": "PayPeriodStartDate",
    "PayPeriodEndDate": "PayPeriodEndDate",
    "CurrentPeriodGrossPay": "CurrentPeriodGrossPay",
    "YearToDateGrossPay": "YearToDateGrossPay",
    "CurrentPeriodTaxes": "CurrentPeriodTaxes",
    "YearToDateTaxes": "YearToDateTaxes",
    "CurrentPeriodDeductions": "CurrentPeriodDeductions",
    "YearToDateDeductions": "YearToDateDeductions",
    "CurrentPeriodNetPay": "CurrentPeriodNetPay",
    "YearToDateNetPay": "YearToDateNetPay",

    # LLM aliases:
    "TotalHoursWorked": "TotalHours",
    "TotalHours": "TotalHours",
    "AveragePayRate": "AveragePayRate",
    "AvgRate": "AveragePayRate",
    "JobTitle": "JobTitle",
    "Title": "JobTitle",
}

# Which fields are expected only from LLM for now.
LLM_ONLY_FIELDS = {"TotalHours", "AveragePayRate", "JobTitle"}

# Which fields are Python-computed (for now left null).
PYTHON_COMPUTED_FIELDS = {"PayFrequency"}



---------------------------



# employment_module/paystub_normalization.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

from employment_module.config import (
    PAYSTUB_FIELD_ORDER,
    PAYSTUB_FIELD_TYPES,
    PAYSTUB_ALIASES,
    LLM_ONLY_FIELDS,
    PYTHON_COMPUTED_FIELDS,
)


# ---------- basic helpers ----------

def empty_paystub_block() -> Dict[str, Dict[str, Any]]:
    """Return an empty 20-field block: every field present, all null."""
    return {
        field: {"value": None, "confidence": 0}
        for field in PAYSTUB_FIELD_ORDER
    }


def _canon_name(raw_name: str) -> str | None:
    """Map DI/LLM key to canonical name, or None if we don't care about it."""
    canon = PAYSTUB_ALIASES.get(raw_name, raw_name)
    return canon if canon in PAYSTUB_FIELD_ORDER else None


def _normalize_number(v: Any) -> float | None:
    if v is None:
        return None
    try:
        # strip things like "$", ",", spaces
        if isinstance(v, str):
            v = v.replace("$", "").replace(",", "").strip()
        return float(v)
    except (ValueError, TypeError):
        return None


def _normalize_string(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _normalize_date(v: Any) -> str | None:
    """Return ISO date string (YYYY-MM-DD) or None.

    DI already tends to return ISO dates; this is defensive.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # If it's already ISO-ish, just return.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s[:10], fmt)
            return dt.date().isoformat()
        except ValueError:
            continue

    # Last resort: don't break; just return original string.
    return s


def _normalize_value(field: str, v: Any) -> Any:
    t = PAYSTUB_FIELD_TYPES.get(field, "string")
    if t == "number":
        return _normalize_number(v)
    if t == "date":
        return _normalize_date(v)
    # string / default
    return _normalize_string(v)


def _normalize_confidence(conf: Any) -> float:
    """Return confidence as 0–100 float."""
    try:
        c = float(conf)
    except (ValueError, TypeError):
        return 0.0
    # If it's in 0–1 range, convert to percent.
    if 0.0 <= c <= 1.0:
        c *= 100.0
    # Clamp
    if c < 0:
        c = 0.0
    if c > 100:
        c = 100.0
    return round(c, 2)


# ---------- apply DI / LLM / Python ----------

def apply_di_fields(
    block: Dict[str, Dict[str, Any]],
    di_fields: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    di_fields is what you already build in extract_paystub_structured, e.g.:
       {"CurrentPeriodGrossPay": {"value": 2862.96, "confidence": 0.98}, ...}
    """
    if not di_fields:
        return block

    for raw_key, payload in di_fields.items():
        canon = _canon_name(raw_key)
        if canon is None:
            continue

        value = payload.get("value")
        conf = payload.get("confidence", 0)

        norm_value = _normalize_value(canon, value)
        norm_conf = _normalize_confidence(conf)

        if norm_value is None:
            # we still leave it in default null/0 state
            continue

        block[canon] = {
            "value": norm_value,
            "confidence": norm_conf,
        }

    return block


def _extract_llm_value(v: Any) -> Tuple[Any, float | None]:
    """
    Support both:
      {"value": 40.0, "confidence": 95}
    and
      40.0
    from LLM JSON.
    """
    if isinstance(v, dict) and "value" in v:
        value = v.get("value")
        conf = v.get("confidence")
        return value, (float(conf) if conf is not None else None)
    # plain value
    return v, None


def apply_llm_fields(
    block: Dict[str, Dict[str, Any]],
    llm_fields: Dict[str, Any],
    default_llm_conf: float = 80.0,
    di_conf_threshold: float = 60.0,
) -> Dict[str, Dict[str, Any]]:
    """
    llm_fields is JSON from GPT, e.g.:
      {
        "TotalHoursWorked": {"value": 40.0, "confidence": 100},
        "AveragePayRate": {"value": 100.0, "confidence": 100},
        "JobTitle": {"value": "Sales Manager", "confidence": 100}
      }
    or without nested structure.
    """
    if not llm_fields:
        return block

    for raw_key, v in llm_fields.items():
        canon = _canon_name(raw_key)
        if canon is None or canon not in LLM_ONLY_FIELDS:
            continue

        llm_value_raw, llm_conf_raw = _extract_llm_value(v)
        norm_value = _normalize_value(canon, llm_value_raw)
        if norm_value is None:
            continue

        existing = block.get(canon, {"value": None, "confidence": 0})
        existing_value = existing.get("value")
        existing_conf = float(existing.get("confidence") or 0)

        # Only overwrite if:
        #  - existing is null OR
        #  - existing confidence is low
        if existing_value is None or existing_conf < di_conf_threshold:
            conf = (
                _normalize_confidence(llm_conf_raw)
                if llm_conf_raw is not None
                else default_llm_conf
            )
            block[canon] = {
                "value": norm_value,
                "confidence": conf,
            }

    return block


def infer_payfrequency(
    pay_date: Any,
    start_date: Any,
    end_date: Any,
) -> Tuple[str | None, float]:
    """
    Placeholder: later you can infer weekly/biweekly/monthly from date range.
    For now we always return (None, 0).
    """
    return None, 0.0


def apply_payfrequency(
    block: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if "PayFrequency" not in block or "PayFrequency" not in PYTHON_COMPUTED_FIELDS:
        return block

    pay_date = block.get("PayDate", {}).get("value")
    start_date = block.get("PayPeriodStartDate", {}).get("value")
    end_date = block.get("PayPeriodEndDate", {}).get("value")

    freq, conf = infer_payfrequency(pay_date, start_date, end_date)

    # Even if freq is None, we ensure the field exists in the right shape.
    block["PayFrequency"] = {
        "value": freq,
        "confidence": _normalize_confidence(conf),
    }
    return block


# ---------- main entry point ----------

def normalize_paystub(
    di_fields: Dict[str, Dict[str, Any]],
    llm_fields: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Build the final 20-field block for one paystub.
    """
    block = empty_paystub_block()
    block = apply_di_fields(block, di_fields)
    block = apply_llm_fields(block, llm_fields)
    block = apply_payfrequency(block)
    return block


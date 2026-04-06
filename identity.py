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


--------------------------------


I am building a production-grade document processing microservices system on Azure. Please help me estimate monthly and yearly costs and recommend appropriate SKUs.

Architecture overview:

A frontend Web App that receives document upload requests (PDFs).

The Web App sends metadata messages to Azure Service Bus.

PDFs are stored temporarily in Azure Blob Storage (Hot tier).

A classifier Function App is triggered by Service Bus, reads the PDF from Blob, calls Azure AI Document Intelligence (classification), and routes messages to downstream queues based on document type.

Multiple downstream Function Apps (employment verification, identity documents, etc.) process documents from their respective Service Bus queues.

Each processing Function App:

Reads PDF from Blob Storage

Calls Azure AI Document Intelligence (1–3 calls per document)

Writes structured results to Azure SQL Database

Publishes a final message to Service Bus consumed by MuleSoft

Workload assumptions:

Total PDFs per year: ~180,000

Average documents per PDF after classification: 3

Total document processing executions per year: ~540,000

Average PDF size: 10 MB

Average Function execution time: ~30 seconds

Function memory allocation: ~1.5 GB

Processing is queue-based and bursty (batch processing every few minutes, not continuous high traffic)

Azure services used:

Azure App Service (Web App) – Basic or Premium tier

Azure Functions (Consumption Plan)

Azure Service Bus (Standard)

Azure Blob Storage (Hot, LRS)

Azure SQL Database (General Purpose)

Azure AI Document Intelligence

Azure Monitor / Application Insights

Azure Key Vault

Security model:

Managed Identity + RBAC

OAuth-based internal service-to-service access

No Private Endpoints or VNET isolation

What I need:

Recommended SKUs for each service

Monthly and yearly cost estimates

Cost breakdown by:

AI services

Compute (Web App + Function Apps)

Service Bus

Monitoring & logging

Storage (Blob + SQL)

Security (Key Vault, RBAC)

Notes on which components scale linearly with document volume

Please assume East US region and production workload.


--------------------------------------


Community Forms – Azure Costing Summary
1. Purpose of This Document

This document provides a high-level annual cost estimate for processing Community Forms using the existing Azure-based document processing platform.

The costing is derived from the internal Excel pricing calculator and reflects incremental costs only, after removing infrastructure already shared with Paystub and Identity document processing.

2. Scope of Community Forms Processing

The Community Forms solution includes:

Document ingestion

Classification

Custom extraction using Azure AI Document Intelligence

Asynchronous processing using Service Bus

Storage of documents and extracted metadata

Monitoring, logging, and security controls

All processing is executed within the existing microservice architecture.

3. Document Volumes (Annual)
Document Type	Extraction Type	PDFs per Year
Exemption Request Form	Custom	108,000
Medically Frail Certification Form	Custom	24,000
CE Good Cause Form	Custom	108,000
Monthly Activity Form	Custom	54,000
Ongoing Activity Form	Custom	54,000
Total		348,000 PDFs / year

Assumption: Each PDF may contain multiple documents and is processed through classification and custom extraction.

4. Cost Breakdown (Annual)
4.1 Classification and Extraction Costs
Service	Annual Cost (USD)
Classification	5,076
Custom Extraction (Document Intelligence)	24,120
Subtotal	29,196
4.2 Compute Infrastructure Costs

Includes only incremental compute required for Community Forms.

Component	Annual Cost (USD)
Web App (Fixed)	1,200
Azure Function Apps	674.88
Service Bus	118.26
Subtotal	1,993.14
4.3 Monitoring & Logging
Component	Annual Cost (USD)
Azure Monitor (Logs & Metrics)	132.80
4.4 Storage
Component	Annual Cost (USD)
Blob Storage	30
Azure SQL Database	1,392
Subtotal	1,422
4.5 Security
Component	Annual Cost (USD)
Azure Key Vault	60
RBAC	60
Subtotal	120
5. Total Annual Cost (Before Overlap Adjustment)
Category	Cost (USD)
Classification & Extraction	29,196
Compute Infrastructure	1,993.14
Monitoring	132.80
Storage	1,422
Security	120
Total	32,863.90
6. Shared Infrastructure Cost Adjustment

The Community Forms solution reuses existing infrastructure already provisioned for:

Paystub processing

Identity document processing

As a result, the following costs are not double-counted:

Web App baseline capacity

Core Function App runtime

Shared monitoring workspace

Shared security services

Overlap Deduction

Overlapping cost removed: USD 2,133

7. Final Annual Incremental Cost

Final Community Forms Annual Cost (Incremental):

✅ USD 30,730.90 per year
8. Key Assumptions

Costs are based on Pay-As-You-Go pricing in a standard US region.

Azure Function Apps run on Consumption plan.

Average document processing time ≈ 30 seconds per document.

Logs retained for 30 days.

Blobs retained for short-term processing, not long-term archival.

Identity-based access is used (no Private Endpoints).

9. Conclusion

This estimate represents a conservative and defensible annual cost for Community Forms processing, leveraging:

Existing shared infrastructure

Serverless compute

Usage-based AI services

The model ensures cost efficiency while maintaining scalability and operational visibility.



    -----------------------------------------------------------------------


    1. Project Overview
1.1 Purpose of the Document

The purpose of this Change Control Document (CCD) is to describe the design, architecture, and implementation details of the Document Intelligence Processing Platform developed to support automated ingestion, classification, extraction, and aggregation of structured data from inbound documents.

This document outlines:

Business objectives

System components

End-to-end processing flows

Architectural decisions

Tools and technologies used

The CCD serves as a formal reference for stakeholders, auditors, architects, and implementation teams.

1.2 Expected Business Outcomes

The Document Intelligence platform is designed to achieve the following outcomes:

Automated document intake with minimal manual intervention

Accurate classification of multi-document PDFs into logical document types

Structured data extraction from complex documents such as:

Employment Verification forms

Pay Stubs

Bank Statements

Identity documents

Improved processing throughput and scalability

Reduced operational burden on case workers and downstream systems

Audit-ready traceability of document processing decisions

1.3 Stakeholders & Project Team

The project involves collaboration across technical, functional, and business teams.

Role	Description
Program Sponsor	Oversees program objectives and delivery
Business Stakeholders	Define document processing requirements
Solution Architect	Defines system architecture and design
Data / AI Engineers	Implement classification and extraction logic
Cloud Engineers	Manage Azure infrastructure and deployment
QA / Testing Team	Validate functional and non-functional requirements
1.4 Scope of the Solution
In Scope

PDF ingestion and processing

Document classification

Page-level document splitting

Domain-specific extraction (Employment, Identity, Bank, Paystub)

Confidence-aware structured output

Fan-out / fan-in orchestration

Azure-native deployment

Out of Scope

Manual data correction workflows

UI-based document review

Third-party downstream integrations beyond structured output delivery

3. Business Requirements
3.1 Business Problem Statement

The organization processes a high volume of inbound documents containing mixed document types bundled into single PDF files. Manual review and data entry are time-consuming, error-prone, and difficult to scale.

There is a need for an automated, reliable, and auditable system to:

Identify document types

Extract key data fields

Preserve traceability across document splits

Support future extensibility

3.2 Functional Requirements
Req ID	Business Requirement
BR-01	The system shall ingest multi-page PDF documents
BR-02	The system shall classify document types automatically
BR-03	The system shall split documents by page ranges
BR-04	The system shall extract structured data per document type
BR-05	The system shall capture confidence scores for extracted fields
BR-06	The system shall support parallel processing
BR-07	The system shall aggregate results deterministically
3.3 Non-Functional Requirements
Category	Requirement
Scalability	Support concurrent document processing
Reliability	Isolate failures at document-split level
Security	Enforce authenticated API access
Observability	Provide traceability via correlation IDs
Maintainability	Modular microservice architecture
Performance	Predictable latency per document
3.4 Compliance & Audit Considerations

All document processing decisions must be traceable

Intermediate outputs must be persisted for audit replay

No destructive modification of original documents

Confidence scores retained for compliance review

7. Document Intelligence Components

This section describes the core components that comprise the Document Intelligence platform and their responsibilities within the overall system.

7.1 Component Overview

The Document Intelligence platform consists of the following major components:

Component	Overview
Ingestion API	Entry point for document intake
Classification Service	Identifies document types
Splitter	Creates logical document splits
Domain Extractors	Extract structured data
Aggregation Engine	Combines split results
Storage Layer	Persists intermediate and final outputs
Messaging Layer	Orchestrates async execution
7.2 Ingestion & Entry Point

The Ingestion Service acts as the system entry point.

Responsibilities:

Accept inbound document payloads

Validate request structure

Generate a parent_uuid for correlation

Persist raw documents to Blob Storage

Emit orchestration events to Service Bus

This component ensures:

Decoupling from downstream processing

Reliable retry behavior

Idempotent request handling

7.3 Classification Component

The Classification Service is responsible for identifying document types within a PDF.

Key characteristics:

Operates on page-level content

Uses Azure Document Intelligence classifiers

Produces classification labels and confidence scores

Outputs structured classification metadata

Classification results drive downstream routing decisions.

7.4 Document Splitting Component

The Splitter isolates individual logical documents from a multi-document PDF.

Design principles:

Page-range based splitting

Immutable original PDF

Deterministic output

Stateless execution

Each split is assigned a unique split_doc_id.

7.5 Domain-Specific Extraction Components

Separate extraction services are implemented per document domain:

Domain	Extractor
Employment	Employment Verification Extractor
Identity	Identity Document Extractor
Paystub	Paystub Extractor
Bank	Bank Statement Extractor

Each extractor:

Processes exactly one split document

Produces structured, normalized output

Captures confidence scores

Operates independently of other domains

7.6 Aggregation Engine

The Aggregation Engine performs deterministic fan-in of split-level results.

Responsibilities:

Detect completion of all expected splits

Merge structured outputs

Produce final response payload

Ensure ordering and completeness

Aggregation is triggered only when all required splits have completed successfully.

7.7 Messaging & Orchestration Layer

Azure Service Bus is used to:

Enable asynchronous execution

Support parallel processing

Provide fault isolation

Allow retry and DLQ handling

Each message contains sufficient context to enable stateless processing.

7.8 Storage & Persistence Layer
Storage Type	Purpose
Blob Storage	Raw and reconstructed PDFs
SQL Database	Intermediate and final results
Logs	Observability and audit

Storage design supports:

Replayability

Debugging

Regulatory audit requirements

7.9 Security & Access Control

OAuth-based authentication

Azure AD protected APIs

Role-based access where applicable

No public unauthenticated endpoints

7.10 Extensibility Considerations

The architecture supports:

Adding new document types

Plug-and-play extractors

Model upgrades

Policy-driven routing logic

7.11 Design Guarantees

The system guarantees:

No cross-document contamination

Deterministic outputs

Idempotent processing

Failure isolation

Horizontal scalability


import re

ONLY_DASHES = re.compile(r"^[-\s_]+$")

def norm_ssn(item: dict) -> dict:
    raw = squash_spaces((item or {}).get("value"))
    conf = (item or {}).get("confidence")

    if not raw:
        return {"value": None, "confidence": conf}

    # Remove spaces for checks
    s = raw.replace(" ", "")

    # Case 1: just dashes / underscores ( "--", "- -", "___" )
    if ONLY_DASHES.match(s):
        return {"value": None, "confidence": conf}

    # Case 2: length too small to be meaningful
    digits = re.sub(r"\D", "", s)
    if len(digits) == 0:
        return {"value": None, "confidence": conf}

    # Otherwise keep it (masked or real)
    return {"value": raw, "confidence": conf}


def _find_largest_rectangle(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_cnt = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # filter out noise
        if area > best_area and w > 100 and h > 50:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None

    x, y, w, h = cv2.boundingRect(best_cnt)

    # build a REAL rectangle
    return np.array([
        [x, y],           # top-left
        [x + w, y],       # top-right
        [x + w, y + h],   # bottom-right
        [x, y + h]        # bottom-left
    ], dtype=np.float32)




            

import re

PREFIXES = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "rev"
}

SUFFIXES = {
    "jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"
}

def parse_name(full_name: str):
    if not full_name or not full_name.strip():
        return {
            "prefix": None,
            "first_name": None,
            "middle_name": None,
            "last_name": None,
            "suffix": None
        }

    # Normalize
    name = re.sub(r"[.,]", "", full_name.strip())
    parts = name.split()

    prefix = None
    suffix = None

    # Check prefix
    if parts and parts[0].lower() in PREFIXES:
        prefix = parts.pop(0)

    # Check suffix
    if parts and parts[-1].lower() in SUFFIXES:
        suffix = parts.pop(-1)

    first_name = None
    middle_name = None
    last_name = None

    if len(parts) == 1:
        first_name = parts[0]

    elif len(parts) == 2:
        first_name, last_name = parts

    elif len(parts) >= 3:
        first_name = parts[0]
        last_name = parts[-1]
        middle_name = " ".join(parts[1:-1])

    return {
        "prefix": prefix,
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "suffix": suffix
    }

import probablepeople

def parse_name_probable(full_name: str):
    try:
        parsed, _ = probablepeople.parse(full_name)

        out = {}
        for value, label in parsed:
            out[label] = out.get(label, "") + (" " if label in out else "") + value

        return {
            "prefix": out.get("Prefix"),
            "first_name": out.get("GivenName"),
            "middle_name": out.get("MiddleName"),
            "last_name": out.get("Surname"),
            "suffix": out.get("Suffix")
        }

    except probablepeople.RepeatedLabelError:
        return None

from nameparser import HumanName

def parse_name_py(full_name: str):
    name = HumanName(full_name)

    return {
        "prefix": name.title or None,
        "first_name": name.first or None,
        "middle_name": name.middle or None,
        "last_name": name.last or None,
        "suffix": name.suffix or None
    }

import spacy

nlp = spacy.load("en_core_web_sm")

def parse_name_with_nlp(full_name: str):
    doc = nlp(full_name)
    tokens = [t.text for t in doc if not t.is_punct]

    return parse_name(" ".join(tokens))





names = [
    "Dr. John A. Smith Jr.",

]

for n in names:
    print(n, "logic based", "=>", parse_name(n))



for n in names:
    print(n, "=>","Python Module", parse_name_py(n))

for n in names:
    print(n, "=>", "NLP", parse_name_with_nlp(n))



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

EXTRACTION_PROMPT = """
You extract structured data from birth certificates.

Return ONLY valid JSON (no markdown, no explanations, no extra text).
The JSON must contain exactly the following 5 keys.
Each key must map to an object with this shape:
{{
  "value": <string or null>,
  "confidence": <number from 0 to 100>
}}

Keys:
- name
- dob
- gender
- place_of_birth
- citizenship_indication

General Rules:
- If a field is not clearly present or cannot be confidently inferred, set:
  {{ "value": null, "confidence": 0 }}
- Do NOT invent or guess values.
- Use values as written in the document, except for the light normalization rules below.
- Do not include any keys other than the 5 listed.
- Do not include explanations outside the JSON.

Light Normalization Rules (apply only these):

1) gender
- Normalize to exactly: "male", "female", or "other" if the document uses variants like:
  "M", "F", "MALE", "FEMALE", "Boy", "Girl".
- If unclear or not present, use:
  {{ "value": null, "confidence": 0 }}

2) dob
- Return the date exactly as written in the document (no reformatting).
- Look for labels like: "Date of Birth", "DOB", "Born", "Birth Date".

3) name
- Use the child’s full name if available (first + middle + last).
- Prefer names near labels like:
  "Name of Child", "Child’s Name", "Name at Birth".
- Do NOT use parent names.
- If multiple name candidates exist, choose the most official-looking one.

4) place_of_birth
- Prefer city + state (or city + country) if available.
- Otherwise return the most specific location string present.
- Look for labels like:
  "Place of Birth", "City of Birth", "County of Birth", "Born in".

citizenship_indication Rules:
- Use "US citizen" only if BOTH are true:
  - The document is clearly a U.S. birth certificate
    (e.g., issued by a U.S. state or county authority), AND
  - The place of birth is in the United States.
- Use "non-US citizen" only if the document clearly indicates foreign birth.
- Otherwise set:
  {{ "value": null, "confidence": 0 }}

Confidence Scoring Rules (0–100):

- 90–100:
  Field is explicitly labeled and clearly present
  (e.g., "Date of Birth: 01/02/2000").

- 70–89:
  Field is present but with minor ambiguity
  (OCR noise, formatting issues, or multiple close candidates).

- 40–69:
  Field is inferred from layout or context
  but not directly labeled.

- 1–39:
  Very weak or highly ambiguous signal.
  Avoid this range; prefer null instead.

- 0:
  Field not found → value must be null.

Strict Rules:
- If value is null, confidence MUST be 0.
- If confidence > 0, value MUST NOT be null.
- Do NOT output "unknown" or any placeholder strings.
- Do NOT guess.
- Do NOT include explanations outside JSON.

Here is the birth certificate OCR text:
\"\"\"{text}\"\"\"

Return JSON only with the required keys and the required value/confidence structure.
"""

def extract_text_from_ocr(file_bytes: bytes) -> str:
    """
    Extracting the raw text from pdf through OCR
    """
    print("Starting OCR text extraction with prebuilt-read...")

    poller = ocr_client.begin_analyze_document(
        model_id="prebuilt-read",
        body=BytesIO(file_bytes),
        content_type="application/octet-stream",
    )
    result = poller.result()

    text = "\n".join(
        line.content
        for page in result.pages
        for line in (page.lines or [])
        if line.content
    )

    print("OCR text extraction complete. Extracted text preview:")
    print("----- OCR RAW TEXT START -----")
    print(text[:3000])
    print("----- OCR RAW TEXT END -----")

    return text


# def extract_llm_fields(text: str):
#     """
#     Calls Azure OpenAI and prints ONLY what the LLM returns (raw content).
#     Does not parse JSON. Returns the raw string too.
#     """
#     prompt = EXTRACTION_PROMPT.format(text=text[:15000])

#     response = openai_client.chat.completions.create(
#         model=AZURE_OPENAI_DEPLOYMENT,
#         messages=[
#             {"role": "system", "content": "Return JSON only."},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )

#     raw = (response.choices[0].message.content or "").strip()
#     print(raw, flush=True)
#     return raw


# def process_birth_certificate(file_bytes: bytes, filename: str):
#     """
#     Runs OCR -> LLM.
#     Prints ONLY the raw LLM output (nothing else).
#     """
#     text = extract_text_from_ocr(file_bytes)   # keep your existing OCR function as-is
#     return extract_llm_fields(text)


def extract_llm_fields(text: str) -> dict:
    """
    Uses your existing `openai_client` (AzureOpenAI) and EXTRACTION_PROMPT.
    Returns the 5 fields, each as { "value": <str|null>, "confidence": <0-100> }.
    No code normalization; confidence is produced by the LLM.
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:15000])
    print("Sending text to Azure OpenAI LLM for field extraction...")

    def _blank():
        return {"value": None, "confidence": 0}

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "Return JSON only. For each field return {value, confidence}.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        print("RAW LLM RESPONSE:", raw,flush=True)

        parsed = json.loads(raw)

        # Enforce shape + defaults (do NOT alter values)
        out = {
            "name": parsed.get("name", _blank()),
            "dob": parsed.get("dob", _blank()),
            "gender": parsed.get("gender", _blank()),
            "place_of_birth": parsed.get("place_of_birth", _blank()),
            "citizenship_indication": parsed.get("citizenship_indication", _blank()),
        }

        # Defensive cleanup: ensure each field is a dict with value/confidence
        for k in out:
            v = out[k]
            if not isinstance(v, dict):
                out[k] = _blank()
                continue
            if "value" not in v:
                v["value"] = None
            if "confidence" not in v or not isinstance(v["confidence"], (int, float)):
                v["confidence"] = 0
            # clamp confidence to 0..100
            if v["confidence"] < 0:
                v["confidence"] = 0
            if v["confidence"] > 100:
                v["confidence"] = 100

        return out

    except Exception as e:
        print("LLM extraction failed:", e)
        return {
            "name": _blank(),
            "dob": _blank(),
            "gender": _blank(),
            "place_of_birth": _blank(),
            "citizenship_indication": _blank(),
        }

def process_birth_certificate(file_bytes: bytes, filename: str) -> dict:
    """
    Orchestrates:
    1) OCR using extract_text_from_ocr()
    2) LLM extraction using extract_llm_fields()
    3) Returns ONLY the 5 fields with value + confidence
    """

    print(f"Processing birth certificate: {filename}")

    try:
        # Step 1: OCR
        text = extract_text_from_ocr(file_bytes)

        # Step 2: LLM Extraction
        llm_fields = extract_llm_fields(text)

        # Final output: only LLM fields (value + confidence)
        output = {
            "name": llm_fields.get("name"),
            "dob": llm_fields.get("dob"),
            "gender": llm_fields.get("gender"),
            "place_of_birth": llm_fields.get("place_of_birth"),
            "citizenship_indication": llm_fields.get("citizenship_indication"),
        }

        print("Final Birth Certificate Extraction Output:")
        print(json.dumps(output, indent=2))

        return {
            "status": "success",
            "filename": filename,
            "extracted_fields": output,
        }

    except Exception as e:
        print("Birth certificate processing failed:", e)

        return {
            "status": "error",
            "filename": filename,
            "extracted_fields": {
                "name": {"value": None, "confidence": 0},
                "dob": {"value": None, "confidence": 0},
                "gender": {"value": None, "confidence": 0},
                "place_of_birth": {"value": None, "confidence": 0},
                "citizenship_indication": {"value": None, "confidence": 0},
            },
            "message": str(e),
        }
    
if __name__ == "__main__":
    input_file = "Birth_certificate1.jpg"   # change this to test other files

    with open(input_file, "rb") as f:
        file_bytes = f.read()

    result = process_birth_certificate(file_bytes, input_file)

    print("\n===== FINAL JSON OUTPUT =====")
    print(json.dumps(result, indent=2))





quad = _find_largest_rectangle(img)





    ---------------------------------------------





def _confidence_pct(field) -> float:
    # DI confidence is typically 0..1
    return round((getattr(field, "confidence", 0) or 0) * 100, 2)

def _field_value(field):
    """
    Prefer .content (what you see in DI JSON).
    Fallback to typed value fields if content is missing.
    """
    content = getattr(field, "content", None)
    if content not in (None, ""):
        return content

    # Typed fallbacks (SDK varies by version)
    for attr in (
        "valueString", "valueDate", "valueNumber", "valueInteger",
        "valuePhoneNumber", "valueTime", "valueCountryRegion", "valueCurrency"
    ):
        v = getattr(field, attr, None)
        if v not in (None, ""):
            return v

    return None

def _field_to_nested_json(field):
    """
    Recursively converts DI DocumentField (including objects/arrays)
    into your desired {value, confidence} leaf format.
    """
    # OBJECT (like FinalFourPayCheckTable, Week1, etc.)
    value_obj = getattr(field, "valueObject", None) or getattr(field, "value_object", None)
    if value_obj:
        out = {}
        # value_obj is dict[str, DocumentField]
        for k, child in value_obj.items():
            out[str(k)] = _field_to_nested_json(child)
        return out

    # ARRAY (in case DI returns arrays for tables in some models)
    value_arr = getattr(field, "valueArray", None) or getattr(field, "value_array", None)
    if value_arr:
        return [_field_to_nested_json(item) for item in value_arr]

    # LEAF FIELD
    return {
        "value": _field_value(field),
        "confidence": _confidence_pct(field)
    }




     --------------------------------------------------


from __future__ import annotations
from typing import Any, Dict, Optional
import re
from datetime import datetime

# You already have this from your photo
# from nameparser import HumanName
# def parse_name_py(full_name: str): ...

def _empty_field() -> Dict[str, Any]:
    return {"value": None, "confidence": None}

def _as_field(obj: Any) -> Dict[str, Any]:
    """
    Ensures we always return the structure: {"value": ..., "confidence": ...}
    Accepts:
      - {"value": x, "confidence": y}
      - raw strings (rare) -> wraps with confidence None
      - None -> empty field
    """
    if obj is None:
        return _empty_field()
    if isinstance(obj, dict) and ("value" in obj or "confidence" in obj):
        return {
            "value": obj.get("value", None),
            "confidence": obj.get("confidence", None),
        }
    if isinstance(obj, str):
        return {"value": obj.strip() or None, "confidence": None}
    return _empty_field()

def _clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None

def _normalize_gender(v: Optional[str]) -> Optional[str]:
    """
    Normalize gender to one of: Male/Female/Other/Unknown (or None)
    """
    v = _clean_text(v)
    if not v:
        return None
    x = v.lower()
    if x in {"m", "male", "man", "boy"}:
        return "Male"
    if x in {"f", "female", "woman", "girl"}:
        return "Female"
    if x in {"other", "non-binary", "nonbinary", "nb"}:
        return "Other"
    if x in {"unknown", "unk", "n/a", "na"}:
        return "Unknown"
    # If it's some unexpected value, keep cleaned original
    return v

def _normalize_dob(v: Optional[str]) -> Optional[str]:
    """
    Normalize DOB to ISO: YYYY-MM-DD if parseable, else keep cleaned original.
    (Safe approach: try a few common formats.)
    """
    v = _clean_text(v)
    if not v:
        return None

    candidates = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%b %d %Y",     # Jan 02 2020
        "%B %d %Y",     # January 02 2020
        "%d %b %Y",     # 02 Jan 2020
        "%d %B %Y",     # 02 January 2020
    ]

    # Remove commas like "Jan 2, 2020"
    vv = v.replace(",", "")
    for fmt in candidates:
        try:
            dt = datetime.strptime(vv, fmt).date()
            return dt.isoformat()
        except ValueError:
            continue

    return v  # fallback: keep as provided

def normalize_birth_certificate_fields(llm_fields: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Input expected (LLM output):
      {
        "name": {"value": "...", "confidence": 92},
        "dob": {"value": "...", "confidence": 85},
        "gender": {"value": "...", "confidence": 78},
        "place_of_birth": {"value": "...", "confidence": 80},
        "citizenship_indication": {"value": "...", "confidence": 60}
      }

    Output (standardized):
      OriginalFullName, DateOfBirth, Gender, PlaceOfBirth, CitizenshipIndication,
      NamePrefix_C, NameFirstName_C, NameMiddleName_C, NameLastName_C, NameSuffix_C
    """

    # --- Read incoming fields safely ---
    name_f = _as_field(llm_fields.get("name"))
    dob_f = _as_field(llm_fields.get("dob"))
    gender_f = _as_field(llm_fields.get("gender"))
    pob_f = _as_field(llm_fields.get("place_of_birth"))
    citizen_f = _as_field(llm_fields.get("citizenship_indication"))

    # --- Normalize values (keep confidence as-is) ---
    name_val = _clean_text(name_f["value"])
    dob_val = _normalize_dob(dob_f["value"])
    gender_val = _normalize_gender(gender_f["value"])
    pob_val = _clean_text(pob_f["value"])
    citizen_val = _clean_text(citizen_f["value"])

    # Update normalized values back
    name_f["value"] = name_val
    dob_f["value"] = dob_val
    gender_f["value"] = gender_val
    pob_f["value"] = pob_val
    citizen_f["value"] = citizen_val

    # --- Derived name parts (same confidence score as 'name') ---
    name_conf = name_f.get("confidence", None)
    if name_val:
        parts = parse_name_py(name_val)  # returns dict with prefix/first/middle/last/suffix
        prefix = _clean_text(parts.get("prefix"))
        first = _clean_text(parts.get("first_name"))
        middle = _clean_text(parts.get("middle_name"))
        last = _clean_text(parts.get("last_name"))
        suffix = _clean_text(parts.get("suffix"))
    else:
        prefix = first = middle = last = suffix = None
        name_conf = None  # no name => no confidence for derived fields

    # --- Build standardized output ---
    out: Dict[str, Dict[str, Any]] = {
        "OriginalFullName": {"value": name_val, "confidence": name_f.get("confidence", None) if name_val else None},
        "DateOfBirth": {"value": dob_val, "confidence": dob_f.get("confidence", None) if dob_val else None},
        "Gender": {"value": gender_val, "confidence": gender_f.get("confidence", None) if gender_val else None},
        "PlaceOfBirth": {"value": pob_val, "confidence": pob_f.get("confidence", None) if pob_val else None},
        "CitizenshipIndication": {"value": citizen_val, "confidence": citizen_f.get("confidence", None) if citizen_val else None},

        "NamePrefix_C": {"value": prefix, "confidence": name_conf},
        "NameFirstName_C": {"value": first, "confidence": name_conf},
        "NameMiddleName_C": {"value": middle, "confidence": name_conf},
        "NameLastName_C": {"value": last, "confidence": name_conf},
        "NameSuffix_C": {"value": suffix, "confidence": name_conf},
    }

    # Guarantee all keys exist even if llm_fields missing (already done).
    return out


----------------------------------------




# ev_normalization.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional


# =========================
# Helpers (self-contained)
# =========================

_ONLY_DIGITS = re.compile(r"\D+")
# keep digits, dot, comma, minus only
_MONEY_RX = re.compile(r"[^\d\.,-]+")


def squash_spaces(s: Any) -> Optional[str]:
    """Collapse whitespace/newlines; return None if empty."""
    if s is None:
        return None
    txt = " ".join(str(s).split())
    return txt or None


def to_float(x: Any) -> Optional[float]:
    """
    Parse a number-like value to float.
    Handles '40', '40.0', '40 hours', etc.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = squash_spaces(x)
    if not s:
        return None

    # Try direct float
    try:
        return float(s)
    except Exception:
        pass

    # Extract first numeric token
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def clean_money(s: Any) -> Optional[str]:
    """
    Normalize currency-ish strings:
      "$3, 461. 54" -> "$3,461.54"
      "6500" -> "$6500" (keeps as string; can float later if needed)
    """
    s = squash_spaces(s)
    if not s:
        return None

    # remove non money chars except digits, dot, comma, minus
    s = _MONEY_RX.sub("", s)

    # remove spaces
    s = s.replace(" ", "")

    if not s:
        return None

    # prefix '$' if not present
    if not s.startswith("$"):
        s = "$" + s

    return s


def parse_date(s: Any) -> Optional[str]:
    """
    Convert common date formats to YYYY-MM-DD.
    """
    txt = squash_spaces(s)
    if not txt:
        return None
    if len(txt) < 6:
        return None

    fmts = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
        "%m/%d/%y",
        "%m-%d-%y",
        "%d-%b-%Y",  # 12-Jan-2017
        "%d %b %Y",  # 12 Jan 2017
        "%b %d %Y",  # Jan 12 2017
        "%B %d %Y",  # January 12 2017
    ]

    for f in fmts:
        try:
            dt = datetime.strptime(txt, f)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # fallback: extract digits and attempt M/D/Y
    nums = re.findall(r"\d{1,4}", txt)
    if len(nums) >= 3:
        try:
            mm = int(nums[0])
            dd = int(nums[1])
            yy = int(nums[2])
            if yy < 100:
                yy = 2000 + yy
            dt = datetime(year=yy, month=mm, day=dd)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    return None


def titlecase_job(s: Any) -> Optional[str]:
    """Simple title casing preserving short ALLCAPS tokens like CEO."""
    txt = squash_spaces(s)
    if not txt:
        return None
    parts = txt.split()
    out = []
    for w in parts:
        if w.isupper() and len(w) <= 4:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


def digits_only(s: Any) -> Optional[str]:
    """Return digits only; None if empty."""
    if s is None:
        return None
    txt = squash_spaces(s)
    if not txt:
        return None
    d = _ONLY_DIGITS.sub("", txt)
    return d or None


# =========================
# Template builder
# =========================

def build_ev_template() -> Dict[str, Dict[str, Any]]:
    """
    Canonical output format:
      { FieldName: { "value": ..., "confidence": ... }, ... }

    Includes FinalFourPayCheckTable as a {value, confidence} container where value is nested dict.
    """
    def _empty():
        return {"value": None, "confidence": None}

    return {
        "EmployeeName": _empty(),
        "SSN": _empty(),
        "HireDate": _empty(),
        "JobTitle": _empty(),
        "EIN": _empty(),
        "FirstPayCheckDate": _empty(),
        "PayFrequency": _empty(),
        "AverageWorkingHours": _empty(),
        "AvgPay": _empty(),
        "AvgPayFrequency": _empty(),
        "EmploymentType": _empty(),
        "EmploymentEndDate": _empty(),
        "EmploymentEndDateReason": _empty(),
        "FinalPayCheckDate": _empty(),
        "FinalPayCheckAmt": _empty(),
        "CompanyName": _empty(),
        "CompanyAddress": _empty(),

        # Nested table (value will be dict like {"Week1": {...}, ...})
        "FinalFourPayCheckTable": _empty(),
    }


# =========================
# Leaf normalizers (top-level)
# =========================

def _norm_keep_conf(item: dict, new_value: Any) -> dict:
    return {"value": new_value, "confidence": (item or {}).get("confidence")}


def norm_text(item: dict) -> dict:
    v = squash_spaces((item or {}).get("value"))
    return _norm_keep_conf(item, v)


def norm_job_title(item: dict) -> dict:
    v = titlecase_job((item or {}).get("value"))
    return _norm_keep_conf(item, v)


def norm_date(item: dict) -> dict:
    v = parse_date((item or {}).get("value"))
    return _norm_keep_conf(item, v)


def norm_money(item: dict) -> dict:
    v = clean_money((item or {}).get("value"))
    return _norm_keep_conf(item, v)


def norm_hours(item: dict) -> dict:
    v = to_float((item or {}).get("value"))
    return _norm_keep_conf(item, v)


def norm_ein(item: dict) -> dict:
    d = digits_only((item or {}).get("value"))
    # EIN should be 9 digits. If not, set None (keep confidence though).
    if not d or len(d) != 9:
        return _norm_keep_conf(item, None)
    return _norm_keep_conf(item, d)


def norm_ssn(item: dict) -> dict:
    raw = squash_spaces((item or {}).get("value"))
    # allow masked or digits; validate later if needed
    return _norm_keep_conf(item, raw)


FIELD_NORMALIZERS = {
    "EmployeeName": norm_text,
    "CompanyName": norm_text,
    "CompanyAddress": norm_text,

    "SSN": norm_ssn,
    "EIN": norm_ein,

    "HireDate": norm_date,
    "EmploymentEndDate": norm_date,
    "FirstPayCheckDate": norm_date,
    "FinalPayCheckDate": norm_date,

    "JobTitle": norm_job_title,

    "AverageWorkingHours": norm_hours,

    "AvgPay": norm_money,
    "FinalPayCheckAmt": norm_money,

    # These are often already clean strings; keep as text
    "PayFrequency": norm_text,
    "AvgPayFrequency": norm_text,
    "EmploymentType": norm_text,
    "EmploymentEndDateReason": norm_text,
}


# =========================
# FinalFourPayCheckTable normalizer (nested)
# =========================

def _leaf(item: Any) -> dict:
    """
    Ensure leaf is {"value":..., "confidence":...}
    """
    if isinstance(item, dict) and "value" in item and "confidence" in item:
        return item
    if isinstance(item, dict) and "value" in item:
        return {"value": item.get("value"), "confidence": item.get("confidence")}
    return {"value": item, "confidence": None}


def _norm_leaf(item: Any, fn) -> dict:
    it = _leaf(item)
    v = fn(it.get("value"))
    return {"value": v, "confidence": it.get("confidence")}


def normalize_final_four_paycheck_table(table_obj: Any) -> Any:
    """
    table_obj from your adaptor is usually:
      {
        "Week1": {"ActualDatePaid": {"value": "...", "confidence": ...}, ...},
        "Week2": {...},
        ...
      }

    Returns same shape, but with normalized leaf values.
    Keeps confidence at each leaf.
    """
    if not isinstance(table_obj, dict):
        return table_obj

    # Keys inside each week (match your DI model keys exactly)
    DATE_KEYS = {"ActualDatePaid"}
    HOURS_KEYS = {"NumOfHours", "Num of Hours", "Num_Of_Hours"}  # include variations if any
    MONEY_KEYS = {"GrossWages", "EITC", "Tips", "Bonus", "Commission"}

    out: Dict[str, Any] = {}

    for week_key, week_obj in table_obj.items():
        if not isinstance(week_obj, dict):
            out[week_key] = week_obj
            continue

        w_out: Dict[str, Any] = {}
        for k, item in week_obj.items():
            if k in DATE_KEYS:
                w_out[k] = _norm_leaf(item, parse_date)
            elif k in HOURS_KEYS:
                w_out[k] = _norm_leaf(item, to_float)
            elif k in MONEY_KEYS:
                w_out[k] = _norm_leaf(item, clean_money)
            else:
                # default: just squash text (keeps confidence)
                w_out[k] = _norm_leaf(item, squash_spaces)

        out[week_key] = w_out

    return out


# =========================
# DI key mapping (DI -> canonical)
# =========================

# If DI keys already match your canonical template keys, keep as-is.
# If your DI model uses different field names, map them here.
EV_KEY_MAP = {
    "EmployeeName": "EmployeeName",
    "SSN": "SSN",
    "HireDate": "HireDate",
    "JobTitle": "JobTitle",
    "EIN": "EIN",
    "FirstPayCheckDate": "FirstPayCheckDate",
    "PayFrequency": "PayFrequency",
    "AverageWorkingHours": "AverageWorkingHours",
    "AvgPay": "AvgPay",
    "AvgPayFrequency": "AvgPayFrequency",
    "EmploymentType": "EmploymentType",
    "EmploymentEndDate": "EmploymentEndDate",
    "EmploymentEndDateReason": "EmploymentEndDateReason",
    "FinalPayCheckDate": "FinalPayCheckDate",
    "FinalPayCheckAmt": "FinalPayCheckAmt",
    "CompanyName": "CompanyName",
    "CompanyAddress": "CompanyAddress",
    "FinalFourPayCheckTable": "FinalFourPayCheckTable",
}


# =========================
# Main normalization entry
# =========================

def normalize_ev(structured_di: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    structured_di expected (from your ev_adaptor) format:

      {
        "EmployeeName": {"value": "...", "confidence": ...},
        ...
        "FinalFourPayCheckTable": {
            "Week1": {"ActualDatePaid": {"value": "...", "confidence": ...}, ...},
            ...
        }
      }

    Returns canonical template with normalized values.
    """
    out = build_ev_template()
    structured_di = structured_di or {}

    for di_key, obj in structured_di.items():
        canonical = EV_KEY_MAP.get(di_key)
        if not canonical or canonical not in out:
            continue

        # Special-case nested table because adaptor gives it as dict-of-weeks (not {value,confidence})
        if canonical == "FinalFourPayCheckTable" and isinstance(obj, dict) and "value" not in obj:
            out[canonical] = {
                "value": normalize_final_four_paycheck_table(obj),
                "confidence": None,  # no single confidence for whole object
            }
            continue

        # Normal leaf field: expected {"value":..., "confidence":...}
        if isinstance(obj, dict) and ("value" in obj or "confidence" in obj):
            fn = FIELD_NORMALIZERS.get(canonical)
            out[canonical] = fn(obj) if fn else {
                "value": (obj or {}).get("value"),
                "confidence": (obj or {}).get("confidence"),
            }
        else:
            # Unexpected shape: keep as value, no confidence
            out[canonical] = {"value": obj, "confidence": None}

    return out


------------------------------------------------



def norm_money(item: dict) -> dict:
    v = clean_money((item or {}).get("value"))
    return {"value": v, "confidence": (item or {}).get("confidence")}

def norm_table_week(week_obj: dict) -> dict:
    """
    week_obj format:
      {
        "ActualDatePaid": {"value": "...", "confidence": ...},
        "GrossWages": {"value": "...", "confidence": ...},
        ...
      }
    returns same structure but normalized values.
    """
    if not isinstance(week_obj, dict):
        return {}

    # per-field normalizers inside each week
    WEEK_FIELD_NORMALIZERS = {
        "ActualDatePaid": norm_date,
        "GrossWages": norm_money,
        "EITC": norm_money,
        "NumOfHours": norm_hours,     # uses to_float()
        "Tips": norm_money,
        "Bonus": norm_money,
        "Commission": norm_money,
    }

    out = {}
    for k, v in week_obj.items():
        fn = WEEK_FIELD_NORMALIZERS.get(k)
        if fn:
            out[k] = fn(v if isinstance(v, dict) else {"value": v, "confidence": None})
        else:
            # default behavior: preserve structure
            # if it's leaf -> pass through (or squash text), if nested -> keep as-is
            if isinstance(v, dict) and ("value" in v or "confidence" in v):
                out[k] = {"value": squash_spaces(v.get("value")), "confidence": v.get("confidence")}
            elif isinstance(v, dict):
                out[k] = v
            else:
                out[k] = {"value": squash_spaces(v), "confidence": None}

    return out

def norm_final_four_paycheck_table(table_obj: dict) -> dict:
    """
    table_obj format:
      {
        "Week1": {...},
        "Week2": {...},
        "Week3": {...},
        "Week4": {...}
      }
    """
    if not isinstance(table_obj, dict):
        return {}

    out = {}
    for week_key, week_val in table_obj.items():
        if isinstance(week_val, dict):
            out[week_key] = norm_table_week(week_val)
        else:
            out[week_key] = {}
    return out





=-0ugj

fn = FIELD_NORMALIZERS.get(canonical)

if fn:
    out[canonical] = fn(obj)
else:
    out[canonical] = {
        "value": (obj or {}).get("value"),
        "confidence": (obj or {}).get("confidence"),
    }



---------------------



def _field_value(field):
    """
    Prefer structured typed values from DocumentField.
    Fallback to content only if typed values are missing.
    """

    # 1️⃣ Numbers first (most reliable for money/hours)
    for attr in ("valueNumber", "valueInteger", "valueCurrency"):
        v = getattr(field, attr, None)
        if v not in (None, ""):
            return v

    # 2️⃣ Date
    v = getattr(field, "valueDate", None)
    if v not in (None, ""):
        return v

    # 3️⃣ String
    v = getattr(field, "valueString", None)
    if v not in (None, ""):
        return v

    # 4️⃣ Fallback to raw OCR text
    content = getattr(field, "content", None)
    if content not in (None, ""):
        return content

    return None







            import os
import json
import base64
from mimetypes import guess_type
from openai import AzureOpenAI

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")   # e.g. https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # your vision-capable deployment name
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"  # works with chat completions examples in Azure docs

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def local_image_to_data_url(image_path: str) -> str:
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_data}"

image_path = r"C:\Users\veboreddy\Desktop\DocAutomation\your_image.jpg"
data_url = local_image_to_data_url(image_path)

prompt = """
Extract the table from this image and return ONLY valid JSON.

Schema:
{
  "table_name": "wages_received",
  "headers": ["Actual Date Paid", "Gross Wages", "EITC", "# of Hours", "Tips", "Bonus", "Commission"],
  "rows": [
    {
      "Actual Date Paid": "",
      "Gross Wages": "",
      "EITC": "",
      "# of Hours": "",
      "Tips": "",
      "Bonus": "",
      "Commission": ""
    }
  ]
}

Rules:
- Return JSON only. No markdown.
- Preserve row order exactly as shown in the image.
- If a cell is blank or only has a slash mark, return "".
- Normalize currency values to plain strings without $.
- Keep dates exactly as seen in the image.
"""

response = client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    messages=[
        {"role": "system", "content": "You are a precise document table extraction assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    max_tokens=2000
)

content = response.choices[0].message.content
print(content)

# Optional: parse it
parsed = json.loads(content)
print(json.dumps(parsed, indent=2))







import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def build_empv_prompt(ocr_text: str) -> str:
    return f"""
You are an expert document information extraction system.

Task:
Extract the required Employment Verification fields from the OCR text and return them in structured JSON.

Instructions:
1. Use only the OCR text provided.
2. Do not guess or infer values that are not explicitly present.
3. If a field is missing, unclear, blank, or not found, return:
   {{
     "value": null,
     "confidence": null
   }}
4. Preserve the original meaning of values.
5. Normalize dates to YYYY-MM-DD only if the OCR text clearly supports the date.
6. Normalize numeric amounts as numbers without currency symbols or commas if clearly present.
7. For text fields, return clean text without extra spaces.
8. For repeated paycheck rows, map them into "Final Four Paychecks" in order: Week1, Week2, Week3, Week4.
9. Do not move values between rows.
10. Treat each paycheck row as one independent record. Never assign a value from one row to another row.
11. Return ONLY valid JSON. No markdown. No explanation.

Output schema:
{{
  "EmployeeName": {{"value": null, "confidence": null}},
  "SSN": {{"value": null, "confidence": null}},
  "HireDate": {{"value": null, "confidence": null}},
  "JobTitle": {{"value": null, "confidence": null}},
  "EIN": {{"value": null, "confidence": null}},
  "FirstPayCheckDate": {{"value": null, "confidence": null}},
  "EmployerName": {{"value": null, "confidence": null}},
  "CompanyAddress": {{"value": null, "confidence": null}},
  "AverageWeeklyHours": {{"value": null, "confidence": null}},
  "PayRate": {{"value": null, "confidence": null}},
  "EmploymentEndDate": {{"value": null, "confidence": null}},
  "FinalPayDate": {{"value": null, "confidence": null}},
  "FinalPayAmount": {{"value": null, "confidence": null}},
  "FinalFourPaychecks": {{
    "Week1": {{
      "ActualDatePaid": {{"value": null, "confidence": null}},
      "GrossWages": {{"value": null, "confidence": null}},
      "EITC": {{"value": null, "confidence": null}},
      "Hours": {{"value": null, "confidence": null}},
      "Tips": {{"value": null, "confidence": null}},
      "Bonus": {{"value": null, "confidence": null}},
      "Commission": {{"value": null, "confidence": null}}
    }},
    "Week2": {{
      "ActualDatePaid": {{"value": null, "confidence": null}},
      "GrossWages": {{"value": null, "confidence": null}},
      "EITC": {{"value": null, "confidence": null}},
      "Hours": {{"value": null, "confidence": null}},
      "Tips": {{"value": null, "confidence": null}},
      "Bonus": {{"value": null, "confidence": null}},
      "Commission": {{"value": null, "confidence": null}}
    }},
    "Week3": {{
      "ActualDatePaid": {{"value": null, "confidence": null}},
      "GrossWages": {{"value": null, "confidence": null}},
      "EITC": {{"value": null, "confidence": null}},
      "Hours": {{"value": null, "confidence": null}},
      "Tips": {{"value": null, "confidence": null}},
      "Bonus": {{"value": null, "confidence": null}},
      "Commission": {{"value": null, "confidence": null}}
    }},
    "Week4": {{
      "ActualDatePaid": {{"value": null, "confidence": null}},
      "GrossWages": {{"value": null, "confidence": null}},
      "EITC": {{"value": null, "confidence": null}},
      "Hours": {{"value": null, "confidence": null}},
      "Tips": {{"value": null, "confidence": null}},
      "Bonus": {{"value": null, "confidence": null}},
      "Commission": {{"value": null, "confidence": null}}
    }}
  }}
}}

Confidence rules:
- confidence must be a number from 0 to 100
- assign high confidence only when the OCR text clearly matches the field
- assign lower confidence if the OCR text is ambiguous or noisy
- if value is null, confidence must also be null

OCR text:
{ocr_text}
""".strip()


def extract_empv_fields_from_ocr(ocr_text: str) -> dict:
    prompt = build_empv_prompt(ocr_text)

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You extract Employment Verification fields from OCR text into strict JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    # Remove accidental markdown fences if model adds them
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    return json.loads(content)


if __name__ == "__main__":
    # Option 1: OCR text from a file
    with open("ocr_text.txt", "r", encoding="utf-8") as f:
        ocr_text = f.read()

    # Option 2: if you already have OCR text in a variable, use that instead
    # ocr_text = "your OCR text here"

    result = extract_empv_fields_from_ocr(ocr_text)

    print(json.dumps(result, indent=2))





1. Roles and responsibilities
1. Machine Learning & Generative AI Model Development
Design, train, configure, and optimize machine learning and generative AI models within enterprise AI platforms, including transformer-based language models and computer vision architectures, to perform automated document classification, structured data extraction, and intelligent processing across large-scale structured and unstructured datasets.

2. Custom Model Training using Enterprise AI Platforms
Lead the design and execution of controlled experiments across multiple model configurations. Interpret quantitative findings and exercise independent technical judgment to determine optimal model selection and recommend implementation strategies.

3. Experimental Design & Statistical Model Evaluation
Design and conduct controlled experiments across multiple model architectures and configurations. Apply statistical performance metrics such as precision, recall, F1-score, ROC-AUC, confidence threshold analysis, and error distribution analysis to determine optimal model selection, interpret quantitative results, and recommend model improvements based on empirical findings.

4. Retrieval-Augmented Generation (RAG) System 
Design and optimize Retrieval-Augmented Generation pipelines utilizing embedding-based vector similarity search to enhance contextual retrieval, grounding accuracy, and response relevance within document intelligence workflows.

5. Computer Vision & OCR Model Benchmarking
Research, benchmark, and implement advanced computer vision and OCR techniques to improve document image preprocessing and extraction accuracy for hand written text date. Perform comparative performance analysis across model configurations and document variations.
6. Model Monitoring, Drift Analysis & Retraining
Oversee model performance monitoring and drift analysis, assess long-term performance trends, and determine retraining strategies to maintain model reliability and automation effectiveness.
7. Data Preparation & Feature Engineering
Perform advanced data preprocessing and normalization of structured and unstructured document data, including Natural Language Processing (NLP) techniques such as tokenization, text cleaning, embedding generation, and contextual text transformation. Conduct feature extraction and transformation of model outputs into structured formats suitable for downstream analytical systems and secure database integration.

8. Algorithmic Optimization & Error Analysis
Analyze model outputs and error patterns using quantitative and statistical techniques to refine algorithmic configurations, improve extraction accuracy, and enhance system performance.

9. AI Solution Integration & Secure Inference Enablement
Support integration of trained AI models into enterprise systems by developing secure REST-based inference endpoints and serverless processing functions. Assist in implementing authentication mechanisms and event-driven orchestration (e.g., message queue-triggered workflows) to enable reliable and scalable AI processing pipelines.

10. Cloud-Based AI Model Lifecycle Management
Leverage enterprise AI platforms such as Azure AI services and AWS AI/ML services to manage model training, validation, deployment, and lifecycle monitoring within secure and scalable cloud environments.




11. Cross-Functional Technical Collaboration
Provide technical guidance to cross-functional teams on AI model feasibility, performance trade-offs, and optimization strategies. Recommend data-driven improvements to enhance document automation efficiency.



def _retry(fn, *, op_name: str, retries: int = AZURE_RETRY_COUNT, delay_sec: int = AZURE_RETRY_DELAY_SEC):
    """
    Retry helper for transient Azure operations.
    Retries with exponential backoff.
    """
    last_ex = None
    wait = delay_sec

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as ex:
            last_ex = ex
            log.warning("%s failed on attempt %s/%s: %s", op_name, attempt, retries, ex)
            if attempt == retries:
                break
            time.sleep(wait)
            wait *= 2

    raise last_ex
                                                                                                          
def _sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash for document content.
    """
    return hashlib.sha256(data).hexdigest()

def _resolve_blob_path(uuid: str) -> str:
    """
    Use deterministic blob path so retries do not create duplicate files.
    """
    return f"{INGEST_PREFIX}/{uuid}.pdf"

def _get_blob_client(uuid: str):
    """
    Returns blob client and blob path for the given UUID.
    """
    blob_path = _resolve_blob_path(uuid)
    return ingest_container.get_blob_client(blob_path), blob_path

def _blob_exists(blob_client) -> bool:
    """
    Check whether blob already exists.
    """
    return blob_client.exists()

def _get_blob_metadata(blob_client) -> dict:
    """
    Fetch metadata for existing blob.
    """
    props = blob_client.get_blob_properties()
    return props.metadata or {}

def _set_blob_metadata(blob_client, metadata: dict) -> None:
    """
    Replace blob metadata with cleaned string values.
    """
    clean_meta = {str(k): str(v) for k, v in metadata.items() if v is not None}
    blob_client.set_blob_metadata(clean_meta)

def _upload_blob_if_needed(uuid: str, no_of_pages: int | None, pdf_bytes: bytes, sha256: str) -> tuple[str, str, dict]:
    """
    Upload the PDF only if it does not already exist.
    Returns blob_path, sha256, metadata.
    """
    blob_client, blob_path = _get_blob_client(uuid)

    if _blob_exists(blob_client):
        metadata = _get_blob_metadata(blob_client)
        log.info("UUID=%s blob already exists at %s", uuid, blob_path)
        return blob_path, sha256, metadata

    metadata = {
        "uuid": uuid,
        "no_of_pages": str(no_of_pages or ""),
        "sha256": sha256,
        "ingest_status": "BLOB_STORED",
        "queue_status": "PENDING",
        "uploaded_at": dt.datetime.utcnow().isoformat() + "Z",
        "last_error": "",
        "last_updated_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    def _do_upload():
        blob_client.upload_blob(
            data=pdf_bytes,
            overwrite=False,
            metadata=metadata,
            content_settings=ContentSettings(content_type="application/pdf"),
        )

    _retry(_do_upload, op_name="blob_upload")

    log.info("UUID=%s blob uploaded at %s", uuid, blob_path)
    return blob_path, sha256, metadata

def _update_queue_status(uuid: str, queue_status: str, last_error: str = "") -> None:
    """
    Update queue processing status in blob metadata.
    """
    blob_client, _ = _get_blob_client(uuid)
    metadata = _get_blob_metadata(blob_client)

    metadata["queue_status"] = queue_status
    metadata["last_error"] = last_error[:200] if last_error else ""
    metadata["last_updated_at"] = dt.datetime.utcnow().isoformat() + "Z"

    _retry(lambda: _set_blob_metadata(blob_client, metadata), op_name="blob_metadata_update")

def _enqueue_sb(uuid: str, no_of_pages: int | None, blob_path: str, sha256: str) -> None:
    """
    Send work message to Service Bus for downstream classifier.
    Uses message_id=uuid so duplicate detection can prevent duplicate messages.
    """
    payload = {
        "UUID": uuid,
        "page_number": no_of_pages,
        "blob_path": blob_path,
        "sha256": sha256,
        "status": "INGESTED",
        "submitted_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    body = json.dumps(payload)

    def _do_send():
        with sb_client:
            sender = sb_client.get_queue_sender(queue_name=SB_QUEUE_CLASSIFY)
            with sender:
                sender.send_messages(
                    ServiceBusMessage(
                        body,
                        content_type="application/json",
                        subject="classify",
                        correlation_id=uuid,
                        message_id=uuid,
                    )
                )

    _retry(_do_send, op_name="service_bus_send")

@app.route(route="ingest", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    """
    Ingest API:
    1. validate request
    2. decode base64
    3. compute hash
    4. upload blob if needed
    5. enqueue service bus if not already published
    6. return structured response
    """

    # --------------------------------
    # Step 1: API key validation
    # --------------------------------
    if API_KEY:
        x_api_key = req.headers.get("x-api-key")
        if not x_api_key or x_api_key != API_KEY:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "message": "Unauthorized"
                }),
                mimetype="application/json",
                status_code=401,
            )

    # --------------------------------
    # Step 2: Parse request
    # --------------------------------
    try:
        body = req.get_json()
        uuid = (body.get("UUID") or "").strip()
        no_of_pages = body.get("no_of_pages")
        content_b64 = body.get("content_b64")

        if not uuid or not content_b64:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "message": "Missing UUID/content_b64"
                }),
                mimetype="application/json",
                status_code=400,
            )

        if len(uuid) > 64:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "UUID": uuid,
                    "message": "UUID too long"
                }),
                mimetype="application/json",
                status_code=400,
            )

    except Exception:
        return func.HttpResponse(
            json.dumps({
                "status": "FAILED",
                "message": "Bad Request"
            }),
            mimetype="application/json",
            status_code=400,
        )

    log.info("UUID=%s request received", uuid)

    # --------------------------------
    # Step 3: Decode and validate file
    # --------------------------------
    try:
        pdf_bytes = _bytes_from_b64(content_b64)
    except ValueError as e:
        err = str(e)

        if err == "TOO_LARGE":
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "UUID": uuid,
                    "message": f"file too large. Max {MAX_UPLOAD_MB} MB"
                }),
                mimetype="application/json",
                status_code=413,
            )

        return func.HttpResponse(
            json.dumps({
                "status": "FAILED",
                "UUID": uuid,
                "message": f"Invalid document payload: {err}"
            }),
            mimetype="application/json",
            status_code=400,
        )

    # --------------------------------
    # Step 4: Compute file hash
    # --------------------------------
    sha256 = _sha256_bytes(pdf_bytes)
    blob_client, blob_path = _get_blob_client(uuid)

    # --------------------------------
    # Step 5: Check if already fully processed
    # --------------------------------
    if _blob_exists(blob_client):
        metadata = _get_blob_metadata(blob_client)

        # Optional hash mismatch check
        existing_sha = metadata.get("sha256")
        if existing_sha and existing_sha != sha256:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "UUID": uuid,
                    "blob_path": blob_path,
                    "message": "UUID already exists with different file content"
                }),
                mimetype="application/json",
                status_code=409,
            )

        if metadata.get("queue_status") == "PUBLISHED":
            log.info("UUID=%s already ingested. blob_path=%s", uuid, blob_path)
            return func.HttpResponse(
                json.dumps({
                    "status": "INGESTED",
                    "UUID": uuid,
                    "blob_path": blob_path,
                    "message": "Already processed"
                }),
                mimetype="application/json",
                status_code=202,
            )

    # --------------------------------
    # Step 6: Upload blob if needed
    # --------------------------------
    try:
        blob_path, sha256, metadata = _upload_blob_if_needed(uuid, no_of_pages, pdf_bytes, sha256)
    except Exception as ex:
        log.exception("Blob upload failed for UUID=%s", uuid)
        return func.HttpResponse(
            json.dumps({
                "status": "FAILED_BLOB",
                "UUID": uuid,
                "message": "Failed to store blob",
                "error": str(ex)
            }),
            mimetype="application/json",
            status_code=503,
        )

    # --------------------------------
    # Step 7: Publish queue message
    # --------------------------------
    try:
        _enqueue_sb(uuid, no_of_pages, blob_path, sha256)
        _update_queue_status(uuid, "PUBLISHED")

        log.info("UUID=%s queued successfully. blob_path=%s", uuid, blob_path)

        return func.HttpResponse(
            json.dumps({
                "status": "INGESTED",
                "UUID": uuid,
                "blob_path": blob_path
            }),
            mimetype="application/json",
            status_code=202,
        )

    except Exception as ex:
        log.exception("Service Bus enqueue failed for UUID=%s", uuid)

        try:
            _update_queue_status(uuid, "FAILED", str(ex))
        except Exception:
            log.exception("Failed to update blob metadata after queue failure for UUID=%s", uuid)

        return func.HttpResponse(
            json.dumps({
                "status": "FAILED_QUEUE",
                "UUID": uuid,
                "blob_path": blob_path,
                "message": "Failed to enqueue message",
                "error": str(ex)
            }),
            mimetype="application/json",
            status_code=503,
        )
                                                                                                          
5.2.7 Initial Orchestration Event Emission

After the document is durably persisted in Blob Storage, the ingestion service emits an event to Azure Service Bus to trigger downstream asynchronous processing.

This marks the transition from the synchronous ingestion layer to the asynchronous document processing pipeline.

The event is published only after:

request validation is successful
Blob persistence is successful
required Blob metadata is recorded

This ensures downstream processing is never started without first having a durable copy of the original document.

5.2.7.1 Queue Publication

The ingestion service publishes a message to the classification queue in Azure Service Bus.

The purpose of this message is to notify the classifier service that a new document is available for processing.

The queue message does not contain the full document payload.
Instead, it contains the document reference and required metadata so downstream services can retrieve the source document from Blob Storage.

This keeps the event payload lightweight and avoids passing large binary content through the queue.

5.2.7.2 Event Payload Structure

The queue message includes the minimum information required to start downstream processing.

Logical fields included:

UUID
Blob path
Page count (if available)
File hash (SHA-256)
Status
Submission timestamp

This event acts as the single trigger for the next stage of the workflow.

Sample Queue Event
{
  "UUID": "fc8da6fd-1cab-4691-b657-5e8cbba5fcf2",
  "page_number": 1,
  "blob_path": "ingest/fc8da6fd-1cab-4691-b657-5e8cbba5fcf2.pdf",
  "sha256": "<sha256-hash>",
  "status": "INGESTED",
  "submitted_at": "2026-03-24T14:25:10Z"
}

👉 Put this sample in a code block, same way you showed sample payload earlier.

5.2.7.3 Service Bus Message Properties

In addition to the message body, the ingestion service sets Service Bus message properties to support traceability and duplicate protection.

Message properties used:

subject = "classify"
correlation_id = UUID
message_id = UUID

The UUID is reused as both the correlation ID and message ID.

This ensures:

end-to-end traceability across services
duplicate detection support in Azure Service Bus
consistent correlation between Blob, logs, and queue event
Message Property Mapping
Property	Value	Purpose
subject	classify	Identifies downstream action
correlation_id	UUID	Enables request tracing
message_id	UUID	Supports duplicate detection

👉 This is a good place for a small table.

5.2.7.4 Duplicate Protection at Queue Level

The queue is configured to use duplicate detection based on the message ID.

Since the ingestion service sets message_id = UUID, repeated publication attempts for the same UUID within the duplicate detection window are treated as duplicates by Azure Service Bus.

This protects downstream services from processing the same ingestion event multiple times.

This is especially important during retry scenarios where:

the original request is retried by the caller
queue publication is retried after a transient failure
the same UUID is submitted again after partial success
5.2.7.5 Publication Retry Behavior

Queue publication is protected by internal retry logic in the ingestion service.

Retry is applied only for transient failures such as:

temporary network interruption
timeout while sending message
temporary Azure Service Bus unavailability

If queue publication succeeds:

Blob metadata is updated with queue_status = PUBLISHED

If queue publication fails after retry attempts:

Blob metadata is updated with queue_status = FAILED
the API returns a failure response to the caller

This allows the same UUID to be retried safely without re-uploading the original file.

5.2.7.6 Partial Success Handling

A partial success scenario may occur when:

Blob upload succeeds
queue publication fails

In this case:

the original document remains safely stored in Blob Storage
the Blob metadata preserves the last known processing state
a subsequent retry uses the same stored document
only the queue publication step is retried

This prevents duplicate Blob uploads and allows the ingestion flow to resume from the last completed step.

5.2.7.7 Emission Guarantee

The ingestion layer guarantees that no orchestration event is emitted before durable Blob persistence is complete.

It also ensures that repeated requests with the same UUID do not create duplicate downstream events when the event has already been successfully published.

This design provides:

reliable transition into asynchronous processing
safe retry behavior
idempotent event emission
5.2.7.8 Event Emission Summary

The event emission flow is as follows:

request is validated
document is stored in Blob Storage
Blob metadata is recorded
Service Bus event is published
queue status is updated in Blob metadata
synchronous acknowledgment is returned to the caller

This completes the ingestion responsibility and hands control to the downstream classification layer.

Where to place tables / examples

Use these in 5.2.7:

Put a code block under:

5.2.7.2 Event Payload Structure

sample queue event JSON
Put a table under:

5.2.7.3 Service Bus Message Properties

subject, correlation_id, message_id
Optional small table under:

5.2.7.5 Publication Retry Behavior
if you want one more table:

Scenario	Behavior
Queue publish succeeds	Update Blob metadata to PUBLISHED
Queue publish fails after retry	Update Blob metadata to FAILED and return error
Same UUID retried later	Reuse Blob and retry queue publication only

That table is optional. If the page is already dense, skip it.

If you want, I’ll draft 5.2.8 Synchronous Response Behavior next in the same style.





Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Ok we need to revamp the tech spec document for our API. It should be in simple language to the point.given screen shots of how the style of write up should be. It should be simple enough to understand use technical terms wherever necessary and avoid sentences with should be very detailed so that even when other person read it who has not coded this should understand. Just keep these in mind. First we will start with ingest process code cleaning then continue with retry and erro handling then documentation.

Got it — I understand the style you want:

Simple, structured, CCD-style

Clear bullets, no long paragraphs

Technical but readable for non-coders

Step-by-step flow

Let’s start with Ingest Process (Code + Cleaning).

1. INGEST PROCESS
1.1 Overview
The ingest process is responsible for receiving documents from the HEIGTHS system, validating them, cleaning the input, and preparing them for downstream processing (classification and extraction).

This process ensures:

Only valid documents are processed

Duplicate or corrupted inputs are filtered

Data is normalized before sending to Azure services

1.2 Input Sources
Documents are received from:

HEIGTHS File System (uploaded/scanned documents)

MuleSoft integration layer

On-prem scheduler (bulk ingestion)

Each document contains:

Document ID (unique identifier)

File (PDF/Image)

Metadata (document type, source system, timestamp)

1.3 Ingest Workflow
Step 1: Document Pickup
Scheduler or API identifies unprocessed documents

Documents are fetched from file system or queue

Status checked from Document Master Table

Step 2: Initial Validation
Each document is validated before processing:

File exists and is accessible

File format is supported (PDF, PNG, JPG)

File size is within limit

Document ID is present

If validation fails:

Mark document as FAILED_VALIDATION

Log error message

Move to error queue

Step 3: Duplicate Check
System ensures idempotency:

Check if Document ID already processed

Check hash of file (optional for deep validation)

If duplicate found:

Skip processing

Log as DUPLICATE_DOCUMENT

No reprocessing triggered

Step 4: File Cleaning & Normalization
This is the code-level cleaning step before sending to AI models.

4.1 File Standardization
Convert all files to standard format (PDF if needed)

Normalize encoding

Ensure consistent DPI for scanned images

4.2 Image Preprocessing (if applicable)
Remove noise (blur, speckles)

Adjust brightness/contrast

Deskew (fix rotated scans)

Crop unnecessary borders

4.3 OCR Pre-Cleaning (Text Preparation)
Remove special/unwanted characters

Normalize whitespace

Fix broken lines (OCR artifacts)

Standardize date formats (if pre-detected)

4.4 Metadata Cleaning
Normalize field names

Ensure required metadata fields exist

Add missing defaults if needed

Example:

doc_type → DOCUMENT_TYPE

emp id → EMPLOYEE_ID

Step 5: Payload Construction
Prepare request for downstream processing:

Convert file to byte array / base64

Attach metadata

Add tracking fields:

Request ID

Timestamp

Source

Step 6: Queue Submission
Push cleaned document to Service Bus Queue

Queue example:

ingest-queue

classification-queue

Message contains:

Document payload

Metadata

Processing flags

Step 7: Status Update
Update database:

Status → INGESTED

Timestamp recorded

Ready for next stage (classification/extraction)

1.4 Logging
Each step logs structured data:

Document ID

Stage (INGEST / CLEANING / VALIDATION)

Status (SUCCESS / FAILED)

Error message (if any)

Logs are sent to:

Application Insights

Central logging system

1.5 Output
After ingest process:

Clean, validated document is available

Document is queued for processing

System is ready for:

Classification

Data Extraction

1.6 Key Design Considerations
Idempotent processing (no duplicates)

Fault isolation (failures do not block pipeline)

Scalable ingestion via queues

Clean input improves AI accuracy

1.7 Example Flow (Simple)
Document uploaded → HEIGTHS

Scheduler picks document

Validate file + metadata

Clean file (image + text)

Build payload

Push to Service Bus

Update status

If this looks good, next I’ll do:

👉 Retry & Error Handling (very important for your system design + production readiness)
— I’ll include DLQ, retry strategy, exponential backoff, idempotency, etc.


Uploaded image
Uploaded image
Uploaded image
Uploaded image
This is current ingest app. How is it for retry and failed calls scenarioa

From these screenshots, the current ingest app is okay for basic failure handling, but it is not strong enough yet for production-grade retry and failed-call scenarios.

What I can see:

Current flow

Validates API key

Reads JSON body

Validates UUID and content_b64

Decodes base64 and checks upload size

Uploads PDF to Blob

Sends message to Service Bus

Returns 202 INGESTED on success

Returns 4xx/5xx on failures

So the app already has basic guardrails, but retry handling is still very limited.

How it behaves today
1. Request validation failures
Examples:

Missing UUID

Missing content_b64

UUID too long

Invalid base64

File too large

Current behavior

Returns 400 or 413

Request stops immediately

No retry is attempted

This is correct
Because these are client/input errors, retry will not help.

2. Blob upload failure
I can see:

_store_blob(...)

wrapped in try/except

on failure returns 503 with "Failed to store blob"

Current behavior

If Blob upload fails, API call fails

Caller has to retry manually

No internal retry inside the app

Risk

Temporary Azure Storage issue

Short network interruption

Token/auth transient issue

Timeout during upload

In these cases, the request may fail even though retrying after a few seconds would have worked.

3. Service Bus enqueue failure
I can see:

_enqueue_sb(...)

wrapped in try/except

on failure returns 503 with "Failed to enqueue message"

Current behavior

Blob may already be uploaded

But queue send fails

API returns failure

Caller may retry the whole request

This is the biggest issue right now

Because on retry:

same file may upload again

same UUID may be reused

same message may be sent again

duplicate processing may happen unless controlled elsewhere

So this flow is vulnerable to partial success:

Blob success

Queue failure

Client retries

Duplicate or inconsistent state possible

What is missing for retry scenarios
1. No explicit retry policy in code
I do not see any:

exponential backoff

fixed retry count

retry only for transient errors

retry for Azure SDK operations

That means failures are handled as:

try once

if fail, return error

This is fine for early dev, but weak for production.

2. No idempotency protection
I do not see a strong duplicate prevention mechanism yet.

You do compute:

sha256

blob_path

UUID

That is good groundwork.

But I do not see:

check whether UUID already exists

check whether blob already uploaded and already queued

check whether message for this UUID was already published

durable ingest status table

Without idempotency, retries can create duplicates.

3. No recovery for partial failure
Most important case:

Scenario
Request received

Blob upload succeeds

Service Bus send fails

API returns 503

Now the system has:

document in Blob

no queue message

caller sees failure

If caller retries:

another upload may happen

another message may happen later

status becomes unclear

This needs a recovery design.

4. No dead-letter or failed-ingest path at API level
I do not see:

failed ingest table

retry queue

poison request handling

quarantine container

manual re-drive flow

Right now failed requests simply return HTTP error.

That means if upstream does not retry correctly, document may be lost from processing flow.

What is already good
1. Clear separation of steps
The code is cleanly split:

decode

store blob

enqueue message

That is good for adding retries later.

2. Correct use of status codes
400 for bad input

401 for unauthorized

413 for too large

503 for downstream dependency failure

202 for accepted

This is good API behavior.

3. Hash generation
sha256 is very useful for:

duplicate detection

troubleshooting

integrity checks

4. Correlation with UUID
Using UUID in payload and correlation_id is good.
This helps trace one document across services.

Main retry/failed-call gaps
Gap 1: Retry is caller-dependent
Today, the client must retry.
The API itself does not appear to retry Blob or Service Bus operations.

Gap 2: Duplicate risk on retry
If caller retries after a partial failure, duplicates may happen.

Gap 3: No transactional safety
Blob upload and Service Bus enqueue are two separate steps.
There is no transaction across them.

Gap 4: No persisted ingest state
I do not see a database record such as:

RECEIVED

BLOB_STORED

QUEUED

FAILED

RETRY_PENDING

Without this, recovery is harder.

Best way to describe current state in tech spec
You can write it like this:

Current Retry and Failure Handling
The current ingest API performs input validation and handles failure at each major processing step. If request validation fails, the API returns an immediate client error response. If Blob Storage upload or Service Bus enqueue fails, the API returns a server error response and stops further processing.

The current implementation does not apply internal retry logic for transient dependency failures. Recovery depends on the calling system resubmitting the request. Because document storage and queue submission are separate operations, partial success can occur. For example, a document may be uploaded successfully while queue publication fails. In this case, a repeated request can lead to duplicate upload or duplicate downstream processing unless additional idempotency checks are implemented.

My assessment
For dev / initial integration
This is acceptable.

For QA / production
This is not enough yet for reliable retry and failed-call handling.

What should be added next
Minimum improvements
Add retry for Blob upload

Add retry for Service Bus send

Retry only for transient errors

Use exponential backoff

Add ingest status persistence

Add idempotency check using UUID or hash

Handle partial success case explicitly

Most important production scenario to fix first
The first thing to fix is:

Blob upload success + Service Bus send failure
Because this is the scenario most likely to create confusion, duplicates, and support issues.

Simple verdict
Right now:

good basic validation

good basic exception handling

weak retry design

weak partial-failure recovery

idempotency not yet strong enough

So I would call it:

“Basic error handling exists, but retry and recovery handling is incomplete for production use.”

Next I can help you write the Retry and Error Handling section in the same CCD/simple style based on this exact code.


How to solve all these issues give plan

Here is the plan to solve it properly without overcomplicating the ingest API.

1. Target design
The ingest API must guarantee these things:

bad input fails immediately

temporary Azure failures retry automatically

duplicate requests do not create duplicate processing

partial success can be recovered

every document has a clear processing state

operations team can see what failed and retry safely

2. Fix order
Do this in phases.

Phase 1 — Make current API safe
First improve the existing code without changing full architecture.

2.1 Add request status tracking
Create one durable ingest status record for each UUID.

Suggested statuses:

RECEIVED

VALIDATION_FAILED

BLOB_STORED

QUEUE_PUBLISHED

FAILED_BLOB

FAILED_QUEUE

COMPLETED_INGEST

Store:

uuid

sha256

blob_path

request_time

status

error_code

error_message

retry_count

last_updated

Best place:

Azure SQL table if your platform already uses SQL

or Cosmos/Table storage if you want lighter design

This is the biggest improvement because it makes recovery possible.

2.2 Add idempotency check before processing
Before decoding and uploading:

Check if UUID already exists.

Cases:

if status = QUEUE_PUBLISHED or COMPLETED_INGEST, return success again

if status = BLOB_STORED, do not upload again; only retry queue publish

if status = FAILED_QUEUE, retry queue publish from saved data

if status = FAILED_BLOB, retry blob upload

This prevents duplicate work.

2.3 Add internal retry for Blob upload
Wrap _store_blob() with retry logic.

Retry only for transient errors such as:

timeout

connection reset

5xx storage errors

throttling

Do not retry for:

bad credentials

invalid container

malformed request

Retry policy:

max 3 attempts

exponential backoff: 2s, 4s, 8s

log every retry attempt

2.4 Add internal retry for Service Bus send
Wrap _enqueue_sb() with same approach.

Retry for:

temporary connection issues

timeout

transient service unavailable

throttling

Do not retry for:

invalid queue name

auth misconfiguration

bad message format

Policy:

max 3 attempts

exponential backoff

log each failure

2.5 Handle partial success explicitly
This is the key fix.

Current risk:

Blob stored

queue send failed

caller retries whole request

New behavior:

after blob upload, save status = BLOB_STORED

try queue send

if queue send succeeds, save QUEUE_PUBLISHED

if queue send fails, do not re-upload blob on next retry

reuse saved blob_path and sha256

So retries continue from last successful step.

3. Phase 2 — Make retries operationally strong
3.1 Add a replay/recovery job
Create a small scheduled job or function.

Purpose:

scan ingest table for records in FAILED_QUEUE or FAILED_BLOB

retry them safely

move permanently failing items to manual review

This removes dependence on upstream caller retries.

Frequency:

every 5 minutes is enough initially

3.2 Add dead-letter / quarantine handling
For items that fail repeatedly:

After retry limit:

mark status = DEAD_LETTERED or MANUAL_REVIEW_REQUIRED

save full reason

optionally move file to failure folder/container

alert support team

This prevents silent loss.

3.3 Add duplicate protection at downstream side too
Even if ingest is fixed, downstream consumers should still protect themselves.

Classifier/extractor should check:

has this UUID already been processed?

has this sha256 already been processed?

If yes:

skip duplicate

log duplicate detection

This gives end-to-end idempotency.

4. Phase 3 — Improve API contract
4.1 Return structured responses
Instead of plain text like "Failed to store blob" return JSON.

Example:

{
  "status": "FAILED_QUEUE",
  "uuid": "12345",
  "error_code": "SERVICE_BUS_UNAVAILABLE",
  "message": "Queue publish failed after retries"
}
For success:

{
  "status": "INGESTED",
  "uuid": "12345",
  "blob_path": "ingest/2026/03/23/12345.pdf"
}
This helps MuleSoft and support teams.

4.2 Add correlation identifiers everywhere
You already use UUID. Keep that as main trace ID.

Log it in:

API request start

validation

blob upload

queue publish

retries

final response

This makes troubleshooting easy.

4.3 Standardize error codes
Use fixed error codes like:

INVALID_REQUEST

INVALID_BASE64

FILE_TOO_LARGE

BLOB_UPLOAD_FAILED

QUEUE_PUBLISH_FAILED

DUPLICATE_REQUEST

RETRY_EXHAUSTED

Do not rely only on exception text.

5. Code structure changes
Refactor current code into clear units.

5.1 Recommended flow
ingest()
  -> authenticate()
  -> parse_request()
  -> validate_request()
  -> get_or_create_ingest_record()
  -> check_existing_status()
  -> decode_content()
  -> upload_blob_with_retry_if_needed()
  -> publish_queue_with_retry_if_needed()
  -> update_final_status()
  -> return structured response
5.2 Add helper functions
Create these helpers:

parse_and_validate_request()

create_ingest_record()

get_existing_ingest_record(uuid)

update_ingest_status(uuid, status, ...)

store_blob_with_retry()

enqueue_with_retry()

is_transient_exception(ex)

build_error_response(...)

This will also make tech spec documentation easier.

6. Suggested retry rules
6.1 Retry matrix
Validation failure
retry? No

response: 400

Invalid base64
retry? No

response: 400

File too large
retry? No

response: 413

Blob timeout / transient storage failure
retry? Yes

response after exhausted retries: 503

Service Bus timeout / transient failure
retry? Yes

response after exhausted retries: 503

Duplicate UUID already processed
retry? No

return existing success state

7. Data model recommendation
Create table: INGEST_REQUEST_TRACKING

Columns:

uuid primary key

sha256

blob_path

page_count

request_status

source_system

request_received_at

last_updated_at

retry_count_blob

retry_count_queue

error_code

error_message

is_duplicate

correlation_id

Optional:

raw request metadata

request headers summary

file size

content type

8. Best practical implementation path
Do not rebuild everything at once.

Sprint 1
add status table

add structured logging

add idempotency check by UUID

save BLOB_STORED before queue send

Sprint 2
add retry wrapper for blob

add retry wrapper for queue send

add standardized error codes/responses

Sprint 3
add replay function for failed records

add dead-letter/manual review path

add dashboard/monitoring

Sprint 4
downstream duplicate checks

operational alerts

support runbook

9. Monitoring plan
Track these metrics:

total ingest requests

validation failures

blob upload failures

queue publish failures

retry attempts

dead-lettered requests

duplicate requests

average ingest response time

Create alerts for:

blob failures spike

queue failures spike

too many FAILED_QUEUE

too many requests stuck in BLOB_STORED

10. What to document in tech spec
For the document, write these sections next:

10.1 Ingest Error Handling
Explain:

validation errors

dependency failures

response behavior

logging

10.2 Retry Strategy
Explain:

retry only for transient dependency failures

exponential backoff

retry limits

no retry for bad request errors

10.3 Idempotency and Duplicate Prevention
Explain:

UUID-based request tracking

status-based resume logic

downstream duplicate check

10.4 Recovery and Replay
Explain:

failed requests stored in tracking table

recovery job retries from last completed step

dead-letter/manual review after retry limit

11. Simplest recommended architecture
Use this model:

Client / MuleSoft
→ Ingest API
→ Ingest Tracking Table
→ Blob Storage
→ Service Bus Queue
→ Classifier / Extractor

And add one more component:

Recovery / Replay Function

That one will clean up most operational problems.

12. Final recommendation
If you ask me what 3 things to do first, do these first:

Top 3 immediate fixes
Create ingest tracking table with statuses

Do not re-upload blob if it already succeeded

Add retry with backoff for Blob and Service Bus

These three will solve most real-world failures.

If you want, next I’ll draft the Retry and Error Handling section in your CCD-style wording based on this exact plan.


Can we do with out crewting a database

Yes, you can, but with tradeoffs.

You do not need a database immediately to improve retry and failed-call handling. You can solve a lot using:

Blob Storage metadata

Blob path conventions

Service Bus duplicate detection

Azure Function / App Insights logs

Optional retry/replay queue

But if you skip a database, you must accept that state tracking will be weaker and recovery will be a little less clean.

Best no-database plan
1. Use UUID as the idempotency key
Treat UUID as the single unique request ID.

Rules:

one UUID = one ingest request

same UUID retried = same document, not a new document

never generate a new UUID during retry

This is the base of everything.

2. Make blob path deterministic
Right now your blob path is date-based:

ingest/{yyyy}/{mm}/{dd}/{UUID}.pdf

This is mostly good, but for retries it should be fully deterministic for the same UUID.

Example:

blob_path = f"ingest/{UUID}.pdf"
Or if you want folders:

blob_path = f"ingest/{UUID[:2]}/{UUID}.pdf"
Why:

retry of same UUID points to same blob

no duplicate blob copies

easy existence check before upload

If you keep date in path, retry on another day may create a second copy unless logic is careful.

3. Check blob existence before upload
Before _store_blob():

check whether blob already exists for this UUID

if yes, reuse it

do not upload again

Also read blob metadata:

sha256

no_of_pages

uploaded_at

So flow becomes:

Case A: blob does not exist
decode file

upload blob

set metadata

continue to queue

Case B: blob already exists
skip upload

continue to queue send only

This alone solves the main partial-failure problem.

4. Store ingest state in blob metadata or blob tags
Since you do not want a DB, use Blob metadata/tags as light state storage.

Example metadata:

uuid

sha256

page_count

ingest_status

queue_status

uploaded_at

last_error_code

last_error_message

Example values:

ingest_status = BLOB_STORED

queue_status = PENDING

later:

queue_status = PUBLISHED

Important note:
Blob metadata is not ideal for frequent updates, but for light state it is workable.

Better split:

blob metadata = file facts

blob tags = searchable processing flags if supported in your setup

5. Add Service Bus duplicate detection
If your Service Bus tier supports duplicate detection, enable it on the queue.

Then send message with:

message_id = UUID

This is very important.

If API retries queue send for same UUID:

Service Bus can ignore duplicates within duplicate detection window

That reduces downstream duplicate processing.

So your message should include:

message_id = UUID

correlation_id = UUID

Right now I can see correlation_id=UUID; add message_id too if not already present.

6. Change the processing order slightly
Use this sequence:

New flow
Validate request

Decode base64

Compute sha256

Resolve deterministic blob path from UUID

Check if blob exists

If not exists, upload blob

Update blob metadata/tag = BLOB_STORED

Send Service Bus message with message_id=UUID

Update blob metadata/tag = QUEUE_PUBLISHED

Return 202

This way, if queue send fails after blob upload:

next retry sees blob already exists

upload is skipped

only queue publish is retried

That is your no-DB recovery model.

Retry plan without database
1. Retry inside API for transient failures
Add internal retry wrappers for:

Blob upload

Service Bus send

Use:

max 3 attempts

exponential backoff

retry only for transient errors

2. Do not retry bad input
No retry for:

invalid API key

bad JSON

missing UUID

invalid base64

file too large

That part remains same.

3. On queue failure, keep blob as source of truth
If queue send fails:

blob already exists

metadata/tag says BLOB_STORED

API returns 503

When caller retries:

system checks blob

finds prior success

skips upload

retries queue publish only

This is the cleanest no-DB solution.

How to handle failed calls without database
Option A — Caller retries
Simplest approach.

If API returns 503, MuleSoft or caller retries with same UUID.

Your API logic:

check blob exists?

yes → skip upload

try queue publish again

This is enough for many systems.

Option B — Add a retry/replay queue
Better option, still no DB.

If queue publish fails after all retries:

send a lightweight message to a retry-ingest queue

include UUID, blob_path, sha256, page_count

a separate retry worker reads from this queue and republishes to main queue

This gives asynchronous recovery without DB.

Flow:

upload blob success

publish main queue fails

publish retry queue succeeds

retry worker handles later

This is stronger than only relying on caller retry.

Option C — Periodic blob scan recovery job
Another no-DB option.

Create scheduled function:

scans blobs in ingest container

looks for metadata/tag:

queue_status = PENDING

or queue_status = FAILED

retries queue publish

This works, but scanning blobs at scale is less efficient than a DB.

Good for moderate volume, not ideal for very high scale.

Recommended no-DB architecture
Minimum version
deterministic blob path

blob existence check

blob metadata/tags for status

Service Bus duplicate detection

retry logic in API

Better version
all of above

plus retry queue for failed queue publication

This gives you strong recovery without introducing a database.

What you lose without a database
Be aware of the limitations.

1. Harder reporting
It is harder to answer:

how many failed today?

how many are stuck?

which retries exhausted?

what is current state of all requests?

Because blob metadata is not as query-friendly as a table.

2. Harder operational dashboards
Support team cannot easily run SQL like:

show all failed ingests

show all queue-pending items

show top error codes

3. Harder multi-step workflow tracking
As more modules come in, DB becomes much more useful.

4. Blob metadata updates are not a perfect workflow engine
Usable, yes. Best long-term solution, no.

My recommendation
If you want fastest fix now
Yes, do it without a database first.

Use:

UUID

deterministic blob path

blob existence check

blob metadata/tags

Service Bus duplicate detection

retry queue or scheduled recovery

That will solve most of your practical issues.

Best implementation plan without DB
Phase 1
make blob path deterministic by UUID

check blob existence before upload

add message_id=UUID on Service Bus message

enable duplicate detection on queue

Phase 2
add retry wrapper for blob upload

add retry wrapper for queue send

store queue_status and ingest_status in blob metadata/tags

Phase 3
add retry queue or scheduled replay function

add better structured logging with UUID everywhere

Very direct answer
Yes, you can avoid creating a database.
For your current ingest app, the best no-DB solution is:

use blob storage as the persisted artifact

use blob metadata/tags as lightweight state

use UUID as idempotency key

use Service Bus duplicate detection to block duplicate messages

retry queue publication safely without re-uploading the blob

This is a valid and practical design.

Next I can help you with a code-level no-database design for your current function, step by step.


Uploaded image
Uploaded image
Uploaded image
Uploaded image
First i want to add all this fucntionality. Suggest changes

Yes. Based on your current function, I would add the functionality in this order so you improve reliability without rewriting the whole app.

1. Main problems in current code
Right now your flow is:

validate request

decode base64

upload blob

enqueue Service Bus

return 202

The gaps are:

blob path is date-based, so retries can create duplicate files

no check whether blob already exists

no internal retry for blob upload

no internal retry for Service Bus send

no message_id for duplicate detection

no persisted lightweight state in blob metadata

partial success case is weak:

blob upload succeeds

queue send fails

next retry may upload again

2. Changes to make first
Change 1: Make blob path deterministic
Your current code uses:

blob_path = f"ingest/{now:%Y/%m/%d}/{UUID}.pdf"
Change it to something like:

blob_path = f"ingest/{UUID}.pdf"
or

blob_path = f"ingest/{UUID[:2]}/{UUID}.pdf"
Why:

same UUID always maps to same blob

retry does not create a second file

easy existence check

This is the first fix.

Change 2: Add blob existence check before upload
Before upload_blob, check whether blob is already there.

Add helper:

def _get_blob_client(uuid: str):
    blob_path = f"ingest/{uuid}.pdf"
    return ingest_container.get_blob_client(blob_path), blob_path
Then:

def _blob_exists(blob_client) -> bool:
    return blob_client.exists()
New logic:

if blob exists, do not upload again

reuse existing blob path

read metadata if needed

continue to queue send

This solves your biggest retry issue.

Change 3: Compute hash before upload
Right now hash is computed inside _store_blob().

Move sha256 calculation earlier so it can be reused whether blob exists or not.

Example:

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
Then in main flow:

decode base64

compute sha256

check blob

if blob exists, compare stored metadata hash if you want

else upload

This gives better duplicate control.

Change 4: Add blob metadata for lightweight state
Since you do not want a database, use metadata.

At minimum store:

metadata = {
    "uuid": UUID,
    "no_of_pages": str(no_of_pages or ""),
    "sha256": sha256,
    "ingest_status": "BLOB_STORED",
    "queue_status": "PENDING"
}
After successful queue publish, update metadata to:

"queue_status": "PUBLISHED"
If queue fails after retries:

"queue_status": "FAILED"
This gives you simple state without DB.

Change 5: Add internal retry wrapper
Add one generic retry helper for Azure operations.

Example idea:

import time

def _retry(operation, *, retries=3, delay=2, op_name="operation"):
    last_ex = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as ex:
            last_ex = ex
            log.warning("%s failed on attempt %s/%s: %s", op_name, attempt, retries, ex)
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise last_ex
Use this for:

blob upload

service bus send

metadata updates if needed

This is one of the biggest improvements.

Change 6: Add message_id=UUID to Service Bus message
Your current code has:

ServiceBusMessage(
    body,
    content_type="application/json",
    subject="classify",
    correlation_id=UUID
)
Add:

message_id=UUID
Final:

ServiceBusMessage(
    body,
    content_type="application/json",
    subject="classify",
    correlation_id=UUID,
    message_id=UUID
)
Why:

if queue duplicate detection is enabled, duplicate messages with same UUID can be ignored

makes retries much safer

Also enable duplicate detection on the Service Bus queue itself.

Change 7: Split upload and metadata handling
Your current _store_blob() both uploads and returns path/hash.

Change it into smaller steps.

Suggested structure
_resolve_blob_path(uuid)

_blob_exists(uuid)

_upload_blob(uuid, pdf_bytes, no_of_pages, sha256)

_set_blob_status(uuid, ingest_status, queue_status, last_error=None)

This will make retry and documentation much cleaner.

3. Suggested new flow
Your ingest function should behave like this:

New flow
Authenticate request

Parse JSON

Validate UUID and content

Decode base64

Compute sha256

Resolve deterministic blob path

Check if blob exists

If not exists:

upload blob with metadata

set ingest_status=BLOB_STORED

If blob exists:

skip upload

If metadata says queue already published:

return 202 immediately

Else try Service Bus send with retry

On success:

update metadata queue_status=PUBLISHED

return 202

On failure:

update metadata queue_status=FAILED

return 503

That is the cleanest no-DB pattern.

4. What to change in each current function
A. Environment settings
Add retry knobs and maybe path mode:

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
AZURE_RETRY_COUNT = int(os.getenv("AZURE_RETRY_COUNT", "3"))
AZURE_RETRY_DELAY_SEC = int(os.getenv("AZURE_RETRY_DELAY_SEC", "2"))
Optional:

INGEST_PREFIX = os.getenv("INGEST_PREFIX", "ingest")
B. _bytes_from_b64()
Current function is okay for basic validation.

Suggested additions:

strip whitespace/newlines before decoding

optionally reject empty bytes

maybe validate MIME separately later if needed

Example improvement:

b64 = (b64 or "").strip()
You can keep the %PDF warning only as warning.

C. _store_blob()
Current issues:

date-based path

always overwrites

no existence check

no state metadata

Suggested changes:

deterministic path

no overwrite by default if using existence logic

metadata includes status

separate metadata update from upload

Instead of always:

overwrite=True
better:

check exists first

upload only if not exists

This avoids accidental replacement.

D. _enqueue_sb()
Add message_id=UUID.

Also include richer payload:

payload = {
    "UUID": UUID,
    "page_number": no_of_pages,
    "blob_path": blob_path,
    "sha256": sha256,
    "status": "INGESTED",
    "submitted_at": dt.datetime.utcnow().isoformat() + "Z"
}
This is fine already. Main changes:

retry wrapper

message_id

maybe log before and after send

E. ingest()
This is where biggest orchestration changes go.

Add these checks:
After decoding and hash generation:

resolve blob path

get blob client

if exists:

fetch metadata

if queue already published, return success

else skip upload and retry queue

This solves:
duplicate upload

safe re-drive after queue failure

idempotent retry

5. Recommended exact behavior for retries
Case 1: Bad request
Examples:

missing UUID

invalid base64

file too large

Behavior:

return 400 or 413

no retry

Case 2: Blob upload transient failure
Behavior:

internal retry 3 times

if still fails, return 503

Case 3: Service Bus transient failure
Behavior:

internal retry 3 times

if still fails, update blob metadata queue_status=FAILED

return 503

Case 4: Same UUID retried after queue failure
Behavior:

blob exists

skip upload

retry queue send only

Case 5: Same UUID retried after full success
Behavior:

blob exists

metadata says published

return 202 without doing anything

6. Important additional fixes
Fix 1: Add structured JSON responses
Instead of plain strings like:

"Failed to enqueue message"
Return:

{
  "status": "FAILED_QUEUE",
  "UUID": uuid,
  "blob_path": blob_path,
  "message": "Failed to enqueue message"
}
And for success:

{
  "status": "INGESTED",
  "UUID": uuid,
  "blob_path": blob_path
}
This is better for MuleSoft and support.

Fix 2: Use consistent logging
Right now logs are okay but improve them.

Add UUID in every log:

request received

blob exists

blob upload start/success/failure

queue send start/success/failure

retry attempt

Example:

log.info("UUID=%s blob already exists at %s", uuid, blob_path)
Fix 3: Update blob metadata after queue result
Need helper like:

def _update_blob_metadata(blob_client, metadata: dict):
    blob_client.set_blob_metadata(metadata)
Use it after:

blob upload success

queue publish success

queue publish failure

7. Best small refactor
I would refactor into these helpers:

_resolve_blob_path(uuid)

_get_blob_client(uuid)

_blob_exists(blob_client)

_get_blob_metadata(blob_client)

_sha256_bytes(pdf_bytes)

_upload_blob(blob_client, pdf_bytes, metadata)

_update_blob_status(blob_client, metadata, ingest_status, queue_status, error_code="")

_retry(...)

_send_to_service_bus(...)

This is enough. No need for big redesign yet.

8. Suggested implementation order
Step 1
Change blob path to deterministic UUID path

Step 2
Add blob existence check

Step 3
Add message_id=UUID

Step 4
Add retry wrapper for upload and queue send

Step 5
Add blob metadata statuses

Step 6
Change ingest() to skip upload on retry and skip queue if already published

This order is practical and low-risk.

9. Minimal pseudocode of desired final behavior
def ingest(req):
    validate request
    pdf_bytes = decode base64
    sha256 = hash(pdf_bytes)

    blob_client, blob_path = _get_blob_client(uuid)

    if blob_client.exists():
        metadata = blob_client.get_blob_properties().metadata
        if metadata.get("queue_status") == "PUBLISHED":
            return success_response(uuid, blob_path, "ALREADY_INGESTED")
    else:
        _retry(lambda: _upload_blob(blob_client, pdf_bytes, metadata), op_name="blob_upload")

    try:
        _retry(lambda: _enqueue_sb(uuid, no_of_pages, blob_path, sha256), op_name="queue_send")
        update metadata queue_status=PUBLISHED
        return success_response(uuid, blob_path, "INGESTED")
    except Exception:
        update metadata queue_status=FAILED
        return failed_response(uuid, blob_path, "FAILED_QUEUE")
That is basically what you want.

10. My direct recommendation for your current code
If you want only the most valuable changes first, do these 5 now:

Change blob path from date-based to UUID-based

Check blob existence before upload

Add retry wrapper around blob upload

Add retry wrapper around Service Bus send

Add message_id=UUID and blob metadata queue_status

These five changes will solve most of the real retry and failed-call issues without adding a database.

Next, I can turn this into a drop-in updated version of your current code structure using the same function names you already have.


Is this more or less costly than creating a database table and calling it in azure. Because we are already using a database we would just be creating another table


Thought for 22s
If you already have an Azure database in this solution, then adding one more table is usually cheaper in practice than building and maintaining the no-DB workaround.

Why:

A Blob-based workaround adds extra storage operations for:

blob existence check

metadata read

metadata update after upload

metadata update after queue success/failure

Azure Blob pricing charges based on the quantity and type of operations, not just storage volume. 

By contrast, if you already pay for an Azure SQL Database, creating one extra table usually does not create a separate fixed monthly charge by itself. Azure SQL Database pricing is primarily driven by the database compute/storage tier you already provisioned, and Microsoft directs you to the pricing calculator for the actual tier-based cost. 

So the practical answer is:

If you need a brand new separate database just for this: no-DB may be cheaper.

If you already have an existing database and would only add another table: the table approach is usually the better value and often the cleaner design.

Also, from an engineering point of view, the DB-table approach gives you:

easier status tracking

easier retries

easier reporting

easier support/debugging

cleaner idempotency handling

The no-DB design saves schema work, but it increases logic complexity and operational effort.

My recommendation for your case:
Use the existing database and add an ingest tracking table.
It is likely the lower total cost once you include developer time, support effort, and production troubleshooting.

A simple table like this is enough:

uuid

sha256

blob_path

status

retry_count

error_code

error_message

created_at

updated_at

If you want, I’ll map your current function code to the exact DB-table based changes step by step.


Uploaded image
Just like this give the business requiremnt for our api

2. BUSINESS REQUIREMENTS
The purpose of this API bundle is to implement the following:

Establish secure connection with Microsoft Azure through MuleSoft

Receive document payload from upstream system through API

Support authentication and request validation before processing

Allow controlled integration with Azure Storage and Azure Service Bus

Accept unprocessed documents from HEIGTHS file system / upstream source

Receive documents in base64 format along with required metadata

Support document level unique identifier (UUID) for tracking

Validate required request fields before processing begins

Validate and clean incoming API request

Validate request structure and mandatory fields

Validate API key or configured authentication mechanism

Validate file size against configured upload limit

Validate base64 content before decoding

Perform basic PDF sanity check before storing document

Store document in Azure Blob Storage

Upload valid document to Azure Blob Storage container

Store document using consistent naming/path convention

Capture document metadata such as:

UUID

number of pages

file hash (sha256)

ingest status

Prevent unnecessary duplicate upload for already processed request

Send ingestion message to Azure Service Bus

Publish message for downstream document classification and extraction

Include required payload details such as:

UUID

blob path

page count

document hash

submitted timestamp

Support correlation of API request and downstream processing

Handle invalid requests with clear API response

Reject requests with missing UUID or missing document content

Reject invalid base64 payload

Reject file that exceeds maximum upload limit

Return appropriate HTTP response code and message for each failure case

Handle dependency failures during storage and queue operations

Detect Blob Storage upload failure

Detect Service Bus enqueue failure

Return failure response when downstream Azure dependency is unavailable

Log failure details for support and retry analysis

Support retry and recovery for failed calls

Allow safe retry for transient Azure failures

Avoid duplicate processing during repeated requests

Resume processing from last successful step where applicable

Support tracking of failed upload and failed queue scenarios

Maintain document processing status

Capture current ingest state for each request

Track whether document is:

received

validated

stored

queued

failed

Provide visibility for troubleshooting and operational support

Provide structured logging and traceability

Log request start and completion

Log validation failures

Log Blob Storage and Service Bus failures

Use UUID as correlation identifier across all steps

Support production monitoring and issue investigation

Return standard API response to calling system

Return success response when document is accepted for downstream processing

Return response with document identifier and blob path

Return failure response with error reason for rejected or failed requests

Support future scalability and production readiness

Support high-volume document ingestion

Support configurable retry settings

Support extension for additional document types and processing stages

Support operational monitoring and reporting

If you want, I’ll next write the Ingest Process section in the same exact style.


Shorten into only bullet points

2. BUSINESS REQUIREMENTS
Establish secure connection with Azure via MuleSoft

Accept documents (base64) with UUID and metadata

Validate request (auth, required fields, size, format)

Decode and perform basic PDF validation

Store document in Azure Blob Storage with metadata

Avoid duplicate uploads for same UUID

Send ingestion message to Azure Service Bus

Include UUID, blob path, hash, and timestamp in message

Handle invalid requests with proper error responses

Handle Blob Storage and Service Bus failures

Support safe retries without duplicate processing

Maintain document ingest status (received, stored, queued, failed)

Log all steps using UUID for traceability

Return structured success/failure API responses

Support scalability and future extensibility


Shorten to 5-6 points

2. BUSINESS REQUIREMENTS
Securely receive and validate documents (UUID, base64, size, auth)

Store documents in Azure Blob Storage with metadata

Avoid duplicate processing using UUID-based handling

Send ingestion message to Azure Service Bus for downstream processing

Handle errors and retries for storage and queue failures

Provide logging, status tracking, and structured API responses


This is for all of thr api not just ingest

2. BUSINESS REQUIREMENTS
Establish secure API integration with Azure via MuleSoft

Ingest, process, and manage documents across classification and extraction workflows

Store documents and extracted data with proper tracking and metadata

Enable communication between services using Azure Service Bus

Handle validation, errors, and retries across all API operations

Provide end-to-end logging, status tracking, and structured responses for all APIs


Explain more clearly with same number of points

2. BUSINESS REQUIREMENTS
Secure integration with Azure via MuleSoft
Ensure all APIs are securely exposed and consumed through MuleSoft with proper authentication and request validation.

End-to-end document ingestion and processing
Support receiving documents, classifying them, and extracting required data for identity and income verification workflows.

Centralized storage of documents and extracted data
Store uploaded documents in Blob Storage and maintain extracted data and metadata for downstream use and UI display.

Asynchronous processing using Azure Service Bus
Enable communication between ingestion, classification, and extraction services using queue-based messaging for scalability.

Robust validation, error handling, and retry mechanisms
Validate inputs, handle failures across all APIs, and support safe retries without duplicate processing.

Comprehensive logging, tracking, and API responses
Provide end-to-end traceability using unique identifiers, maintain processing status, and return structured responses for all API operations.


What are business requirements here. Check the example images and givd in same style of english and simple explainations and to the point

2. BUSINESS REQUIREMENTS
Securely receive API requests
All APIs must accept requests only from authorized systems and validate required inputs before processing.

Process documents end-to-end
System should support full flow from document intake to classification and data extraction.

Store documents and related data
Uploaded files must be stored in Blob Storage and associated data must be available for further processing and UI usage.

Enable communication between services
Different processing steps (ingest, classify, extract) should communicate using Service Bus queues.

Handle failures and support retries
System should handle errors properly and allow safe retries without creating duplicate processing.

Provide clear status and responses
Each API should return clear success or failure responses and allow tracking of document processing status.


Point two explains everthing in api right. Other points are just moot. It should be able to identify and classify documents that one point like that

Got it — you want clean, outcome-focused business requirements, not technical restatements.

Here’s the corrected version in that style 👇

2. BUSINESS REQUIREMENTS
Securely accept and process API requests
System should allow only authorized systems to submit and process requests.

Identify and classify different document types
System should automatically detect and categorize documents (e.g., pay stub, bank statement, ID).

Extract required information from documents
System should capture key fields needed for verification and downstream processing.

Store documents and extracted information
System should save documents and their data for further use and reference.

Ensure reliable processing without duplication
System should handle failures and retries without processing the same document multiple times.

Provide clear status and results for each request
System should allow tracking of document processing and return clear outcomes for each API call.


Uploaded image
Give name for this.

Based on your content (classification, extraction, Azure pipeline, Service Bus, document flow), a clean and appropriate title would be:

Document Intelligence Processing Pipeline
If you want slightly more formal / doc-style options:

Document Intelligence Architecture Overview

Document Processing and Extraction Workflow

End-to-End Document Intelligence Pipeline

Document Classification and Extraction Architecture

👉 Best fit for your doc (simple + clear):
Document Intelligence Processing Pipeline


ok lets do with no sql db. give code and step by step intructions to follow to edit existint code and change in the azure service bus as well


Thought for 23s
Yes — below is the no-SQL version to add first.

This approach uses:

deterministic blob path by UUID

blob metadata as lightweight state

Service Bus duplicate detection with message_id=UUID

safe retry for Blob upload and queue send

Azure Service Bus duplicate detection keeps a history of MessageId values for a configured window and drops duplicates with the same MessageId during that window. Azure Blob Storage supports user-defined metadata, which you can update from Python. The Python ServiceBusMessage supports message_id and correlation_id. 

1. What to change in Azure Service Bus first
Queue change
Enable duplicate detection on the queue used by ingest/classify.

What this gives you
If the same API call retries and sends the same UUID again as message_id, Service Bus will ignore the duplicate message within the configured duplicate detection window. 

Portal steps
Open Azure Portal

Go to your Service Bus Namespace

Open Queues

Select your queue, for example q-employment

Open Properties

Enable Duplicate detection

Set a Duplicate detection history time window

start with 10 minutes or 30 minutes

Save

What to remember in code
When sending the message, set:

message_id=UUID
correlation_id=UUID
2. Code changes you should make in current file
We will change your code in this order:

add retry config

make blob path deterministic

add blob existence check

add blob metadata helpers

add retry helper

change blob upload logic

change Service Bus send logic

change main ingest flow

3. Step-by-step edits
Step 1 — add imports
At the top of the file, make sure these imports exist:

import json
import time
import hashlib
import datetime as dt
import base64
import logging
import os

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.servicebus import ServiceBusClient, ServiceBusMessage
If you already have them, keep only one copy.

Step 2 — add new environment settings
Near your current env settings, add:

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

AZURE_RETRY_COUNT = int(os.getenv("AZURE_RETRY_COUNT", "3"))
AZURE_RETRY_DELAY_SEC = int(os.getenv("AZURE_RETRY_DELAY_SEC", "2"))

INGEST_PREFIX = os.getenv("INGEST_PREFIX", "ingest")
Step 3 — replace current _store_blob() approach with deterministic blob path
Your current code uses date-based path. Replace that with UUID-based path.

Add these helper functions
def _resolve_blob_path(uuid: str) -> str:
    return f"{INGEST_PREFIX}/{uuid}.pdf"


def _get_blob_client(uuid: str):
    blob_path = _resolve_blob_path(uuid)
    return ingest_container.get_blob_client(blob_path), blob_path
Step 4 — add helper to calculate hash
Add:

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
Step 5 — add blob metadata helpers
Add:

def _blob_exists(blob_client) -> bool:
    return blob_client.exists()


def _get_blob_metadata(blob_client) -> dict:
    props = blob_client.get_blob_properties()
    return props.metadata or {}


def _set_blob_metadata(blob_client, metadata: dict) -> None:
    clean_meta = {str(k): str(v) for k, v in metadata.items() if v is not None}
    blob_client.set_blob_metadata(clean_meta)
Azure Blob Storage supports user-defined metadata and Python SDK methods for reading and setting it. 

Step 6 — add generic retry helper
Add this once:

def _retry(fn, *, op_name: str, retries: int = AZURE_RETRY_COUNT, delay_sec: int = AZURE_RETRY_DELAY_SEC):
    last_ex = None
    wait = delay_sec

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as ex:
            last_ex = ex
            log.warning("%s failed on attempt %s/%s: %s", op_name, attempt, retries, ex)
            if attempt == retries:
                break
            time.sleep(wait)
            wait *= 2

    raise last_ex
This is your internal retry for:

blob upload

blob metadata update

queue send

Step 7 — replace _store_blob() with upload-if-needed logic
Remove your current _store_blob() and replace it with this:

def _upload_blob_if_needed(uuid: str, no_of_pages: int | None, pdf_bytes: bytes, sha256: str) -> tuple[str, str, dict]:
    blob_client, blob_path = _get_blob_client(uuid)

    if _blob_exists(blob_client):
        metadata = _get_blob_metadata(blob_client)
        log.info("UUID=%s blob already exists at %s", uuid, blob_path)
        return blob_path, sha256, metadata

    metadata = {
        "uuid": uuid,
        "no_of_pages": str(no_of_pages or ""),
        "sha256": sha256,
        "ingest_status": "BLOB_STORED",
        "queue_status": "PENDING",
        "uploaded_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    def _do_upload():
        blob_client.upload_blob(
            data=pdf_bytes,
            overwrite=False,
            metadata=metadata,
            content_settings=ContentSettings(content_type="application/pdf"),
        )

    _retry(_do_upload, op_name="blob_upload")

    log.info("UUID=%s blob uploaded at %s", uuid, blob_path)
    return blob_path, sha256, metadata
Why this change:

same UUID always goes to same blob path

if retry happens after partial success, upload is skipped

metadata stores lightweight state

Blob upload with metadata and content settings is supported in the Python SDK. 

Step 8 — add helper to update queue status in blob metadata
Add:

def _update_queue_status(uuid: str, queue_status: str, last_error: str = "") -> None:
    blob_client, _ = _get_blob_client(uuid)
    metadata = _get_blob_metadata(blob_client)

    metadata["queue_status"] = queue_status
    metadata["last_error"] = last_error[:200] if last_error else ""
    metadata["last_updated_at"] = dt.datetime.utcnow().isoformat() + "Z"

    _retry(lambda: _set_blob_metadata(blob_client, metadata), op_name="blob_metadata_update")
Step 9 — replace _enqueue_sb() with safe send
Replace your current _enqueue_sb() with this:

def _enqueue_sb(uuid: str, no_of_pages: int | None, blob_path: str, sha256: str) -> None:
    payload = {
        "UUID": uuid,
        "page_number": no_of_pages,
        "blob_path": blob_path,
        "sha256": sha256,
        "status": "INGESTED",
        "submitted_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    body = json.dumps(payload)

    def _do_send():
        with sb_client:
            sender = sb_client.get_queue_sender(queue_name=SB_QUEUE_CLASSIFY)
            with sender:
                sender.send_messages(
                    ServiceBusMessage(
                        body,
                        content_type="application/json",
                        subject="classify",
                        correlation_id=uuid,
                        message_id=uuid,
                    )
                )

    _retry(_do_send, op_name="service_bus_send")
message_id is the key part for duplicate detection. The Service Bus Python message class supports it. 

Step 10 — keep _bytes_from_b64() but slightly improve it
Replace with:

def _bytes_from_b64(b64: str) -> bytes:
    """Decode base64 to bytes; enforce size and basic PDF sanity."""
    try:
        data = base64.b64decode((b64 or "").strip(), validate=True)
    except Exception:
        raise ValueError("INVALID_B64")

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError("TOO_LARGE")

    if not data:
        raise ValueError("EMPTY_FILE")

    if not data.startswith(b"%PDF"):
        log.warning("Uploaded content does not start with %%PDF header; proceeding.")

    return data
Step 11 — replace the ingest() function
Replace your current ingest() with this version:

@app.route(route="ingest", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    if API_KEY:
        x_api_key = req.headers.get("x-api-key")
        if not x_api_key or x_api_key != API_KEY:
            return func.HttpResponse(
                json.dumps({"status": "FAILED", "message": "Unauthorized"}),
                mimetype="application/json",
                status_code=401,
            )

    try:
        body = req.get_json()
        uuid = (body.get("UUID") or "").strip()
        no_of_pages = body.get("no_of_pages")
        content_b64 = body.get("content_b64")

        if not uuid or not content_b64:
            return func.HttpResponse(
                json.dumps({"status": "FAILED", "message": "Missing UUID/content_b64"}),
                mimetype="application/json",
                status_code=400,
            )

        if len(uuid) > 64:
            return func.HttpResponse(
                json.dumps({"status": "FAILED", "message": "UUID too long"}),
                mimetype="application/json",
                status_code=400,
            )
    except Exception:
        return func.HttpResponse(
            json.dumps({"status": "FAILED", "message": "Bad Request"}),
            mimetype="application/json",
            status_code=400,
        )

    try:
        pdf_bytes = _bytes_from_b64(content_b64)
    except ValueError as e:
        err = str(e)
        if err == "TOO_LARGE":
            return func.HttpResponse(
                json.dumps({"status": "FAILED", "UUID": uuid, "message": f"file too large. Max {MAX_UPLOAD_MB} MB"}),
                mimetype="application/json",
                status_code=413,
            )

        return func.HttpResponse(
            json.dumps({"status": "FAILED", "UUID": uuid, "message": f"Invalid document payload: {err}"}),
            mimetype="application/json",
            status_code=400,
        )

    sha256 = _sha256_bytes(pdf_bytes)
    blob_client, blob_path = _get_blob_client(uuid)

    # Case 1: blob already exists and queue already published
    if _blob_exists(blob_client):
        metadata = _get_blob_metadata(blob_client)
        if metadata.get("queue_status") == "PUBLISHED":
            log.info("UUID=%s already ingested. blob_path=%s", uuid, blob_path)
            return func.HttpResponse(
                json.dumps({
                    "status": "INGESTED",
                    "UUID": uuid,
                    "blob_path": blob_path,
                    "message": "Already processed"
                }),
                mimetype="application/json",
                status_code=202,
            )

    # Case 2: upload blob if needed
    try:
        blob_path, sha256, metadata = _upload_blob_if_needed(uuid, no_of_pages, pdf_bytes, sha256)
    except Exception as ex:
        log.exception("Blob upload failed for UUID=%s", uuid)
        return func.HttpResponse(
            json.dumps({
                "status": "FAILED_BLOB",
                "UUID": uuid,
                "message": "Failed to store blob",
                "error": str(ex),
            }),
            mimetype="application/json",
            status_code=503,
        )

    # Case 3: publish queue if not already published
    try:
        _enqueue_sb(uuid, no_of_pages, blob_path, sha256)
        _update_queue_status(uuid, "PUBLISHED")
        log.info("UUID=%s queued successfully. blob_path=%s", uuid, blob_path)

        return func.HttpResponse(
            json.dumps({
                "status": "INGESTED",
                "UUID": uuid,
                "blob_path": blob_path,
            }),
            mimetype="application/json",
            status_code=202,
        )

    except Exception as ex:
        log.exception("Service Bus enqueue failed for UUID=%s", uuid)
        try:
            _update_queue_status(uuid, "FAILED", str(ex))
        except Exception:
            log.exception("Failed to update blob metadata after queue failure for UUID=%s", uuid)

        return func.HttpResponse(
            json.dumps({
                "status": "FAILED_QUEUE",
                "UUID": uuid,
                "blob_path": blob_path,
                "message": "Failed to enqueue message",
                "error": str(ex),
            }),
            mimetype="application/json",
            status_code=503,
        )
4. What this new flow does
On first success
validates request

decodes base64

computes hash

stores blob once

writes metadata:

ingest_status=BLOB_STORED

queue_status=PENDING

sends Service Bus message with message_id=UUID

updates metadata:

queue_status=PUBLISHED

If blob upload succeeds but queue fails
blob remains in storage

metadata becomes queue_status=FAILED

next retry with same UUID:

blob upload is skipped

queue send is retried only

If same request comes again after full success
blob exists

metadata says queue_status=PUBLISHED

API returns 202 without sending duplicate work

5. Small but important Azure checks
Blob container
Use the same container. No portal change needed unless:

the container does not exist

managed identity / access is missing

Service Bus queue
You must confirm:

duplicate detection is enabled

queue name matches SB_QUEUE_CLASSIFY

Duplicate detection window
Choose a window long enough to cover retry scenarios. Microsoft documents that Service Bus duplicate detection tracks MessageId within the configured time window and drops duplicates in that span. 

Start with:

10 minutes for short retries

30 minutes if MuleSoft may retry later

6. Test cases you should run
Test 1 — happy path
Send valid request:

expect 202

blob created

metadata queue_status=PUBLISHED

Test 2 — same UUID again after success
Send same payload again:

expect 202

no new blob

no duplicate queue work

Test 3 — invalid base64
Send bad base64:

expect 400

Test 4 — file too large
Send file over limit:

expect 413

Test 5 — simulate queue failure
Temporarily use wrong queue name or disable access:

blob should upload

API returns 503 FAILED_QUEUE

metadata should show queue_status=FAILED

Test 6 — retry after queue failure
Restore queue config and resend same UUID:

blob should not upload again

queue should publish

metadata should update to PUBLISHED

7. One optional improvement after this
After this works, add a small replay function later that scans blobs with:

queue_status=FAILED

and republishes them.

That is optional. The changes above are enough for the first safe version.

8. Summary of exactly what you change
In Azure
enable duplicate detection on Service Bus queue

set duplicate detection window

In code
change blob path to ingest/{UUID}.pdf

add blob existence check

add blob metadata state

add retry wrapper

set message_id=UUID

skip re-upload if blob already exists

skip re-send if metadata says already published

return structured JSON responses

If you want, next I’ll give you a single cleaned full file by merging this directly into your current code structure.


give entire code onces for each fucntion in tripple quotes give explaination.

Below is the full no-SQL version in pieces, with each function in triple backticks and a short explanation under it.

Use this as a guide to edit your existing file.

1. Environment and client setup
Replace your current settings / client setup block with this.

import os
import json
import time
import base64
import hashlib
import logging
import datetime as dt

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.servicebus import ServiceBusClient, ServiceBusMessage


# ----------------------------
# Settings / Environment
# ----------------------------
APP_NAME = "DocProc Ingestion API"
API_KEY = os.getenv("INGRESS_API_KEY")  # dev-only gate while OAuth is off

STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")              # preferred (Managed Identity)
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")  # dev fallback
STORAGE_CONTAINER = os.getenv("STORAGE_CONTAINER", "ingest")

SB_NAMESPACE = os.getenv("SERVICEBUS_NAMESPACE")                    # preferred (Managed Identity)
SB_CONNECTION_STRING = os.getenv("SB_CONNECTION_STRING")            # dev fallback
SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-employment")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
AZURE_RETRY_COUNT = int(os.getenv("AZURE_RETRY_COUNT", "3"))
AZURE_RETRY_DELAY_SEC = int(os.getenv("AZURE_RETRY_DELAY_SEC", "2"))
INGEST_PREFIX = os.getenv("INGEST_PREFIX", "ingest")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(APP_NAME)

# ----------------------------
# Azure Credentials
# ----------------------------
cred = DefaultAzureCredential(exclude_environment_credential=False)

# ----------------------------
# Blob client
# ----------------------------
if STORAGE_ACCOUNT_URL:
    blob_service = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=cred)
else:
    if not STORAGE_CONNECTION_STRING:
        raise RuntimeError("No storage credentials configured.")
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

ingest_container = blob_service.get_container_client(STORAGE_CONTAINER)

# ----------------------------
# Service Bus client
# ----------------------------
if SB_NAMESPACE:
    sb_client = ServiceBusClient(SB_NAMESPACE, credential=cred, logging_enable=False)
else:
    if not SB_CONNECTION_STRING:
        raise RuntimeError("No Service Bus credentials configured.")
    sb_client = ServiceBusClient.from_connection_string(SB_CONNECTION_STRING, logging_enable=False)

app = func.FunctionApp()
Explanation
This block:

loads all environment variables

connects to Blob Storage

connects to Service Bus

keeps retry settings configurable

supports both Managed Identity and connection string fallback

2. Retry helper
Add this function.

def _retry(fn, *, op_name: str, retries: int = AZURE_RETRY_COUNT, delay_sec: int = AZURE_RETRY_DELAY_SEC):
    """
    Retry helper for transient Azure operations.
    Retries with exponential backoff.
    """
    last_ex = None
    wait = delay_sec

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as ex:
            last_ex = ex
            log.warning("%s failed on attempt %s/%s: %s", op_name, attempt, retries, ex)
            if attempt == retries:
                break
            time.sleep(wait)
            wait *= 2

    raise last_ex
Explanation
This is the common retry wrapper.
Use it for:

blob upload

blob metadata update

service bus send

It retries 3 times by default, with delays like:

2 sec

4 sec

8 sec

3. Base64 decode and validation
Replace your current _bytes_from_b64() with this.

def _bytes_from_b64(b64: str) -> bytes:
    """
    Decode base64 to bytes.
    Enforces upload size and basic PDF sanity.
    """
    try:
        data = base64.b64decode((b64 or "").strip(), validate=True)
    except Exception:
        raise ValueError("INVALID_B64")

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError("TOO_LARGE")

    if not data:
        raise ValueError("EMPTY_FILE")

    if not data.startswith(b"%PDF"):
        log.warning("Uploaded content does not start with %%PDF header; proceeding.")

    return data
Explanation
This function:

decodes base64

checks upload size

rejects empty file

gives a warning if file is not starting with %PDF

It does not stop non-PDF automatically, but logs it.

4. SHA-256 helper
Add this function.

def _sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash for document content.
    """
    return hashlib.sha256(data).hexdigest()
Explanation
This creates a stable file hash.
Useful for:

troubleshooting

duplicate validation

integrity checks

5. Blob path helper
Add this function.

def _resolve_blob_path(uuid: str) -> str:
    """
    Use deterministic blob path so retries do not create duplicate files.
    """
    return f"{INGEST_PREFIX}/{uuid}.pdf"
Explanation
This is very important.
Instead of date-based path, this uses:

ingest/<UUID>.pdf

So the same UUID always points to the same file.

6. Blob client helper
Add this function.

def _get_blob_client(uuid: str):
    """
    Returns blob client and blob path for the given UUID.
    """
    blob_path = _resolve_blob_path(uuid)
    return ingest_container.get_blob_client(blob_path), blob_path
Explanation
This avoids repeating blob path logic in multiple places.

7. Blob existence check
Add this function.

def _blob_exists(blob_client) -> bool:
    """
    Check whether blob already exists.
    """
    return blob_client.exists()
Explanation
This is used before upload.
If blob already exists:

do not upload again

continue from the next step

8. Read blob metadata
Add this function.

def _get_blob_metadata(blob_client) -> dict:
    """
    Fetch metadata for existing blob.
    """
    props = blob_client.get_blob_properties()
    return props.metadata or {}
Explanation
This reads stored lightweight state like:

queue status

hash

number of pages

upload timestamp

9. Update blob metadata
Add this function.

def _set_blob_metadata(blob_client, metadata: dict) -> None:
    """
    Replace blob metadata with cleaned string values.
    """
    clean_meta = {str(k): str(v) for k, v in metadata.items() if v is not None}
    blob_client.set_blob_metadata(clean_meta)
Explanation
Blob metadata values must be strings.
This helper cleans and updates them safely.

10. Upload blob only if needed
Replace your current _store_blob() with this.

def _upload_blob_if_needed(uuid: str, no_of_pages: int | None, pdf_bytes: bytes, sha256: str) -> tuple[str, str, dict]:
    """
    Upload the PDF only if it does not already exist.
    Returns blob_path, sha256, metadata.
    """
    blob_client, blob_path = _get_blob_client(uuid)

    if _blob_exists(blob_client):
        metadata = _get_blob_metadata(blob_client)
        log.info("UUID=%s blob already exists at %s", uuid, blob_path)
        return blob_path, sha256, metadata

    metadata = {
        "uuid": uuid,
        "no_of_pages": str(no_of_pages or ""),
        "sha256": sha256,
        "ingest_status": "BLOB_STORED",
        "queue_status": "PENDING",
        "uploaded_at": dt.datetime.utcnow().isoformat() + "Z",
        "last_error": "",
        "last_updated_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    def _do_upload():
        blob_client.upload_blob(
            data=pdf_bytes,
            overwrite=False,
            metadata=metadata,
            content_settings=ContentSettings(content_type="application/pdf"),
        )

    _retry(_do_upload, op_name="blob_upload")

    log.info("UUID=%s blob uploaded at %s", uuid, blob_path)
    return blob_path, sha256, metadata
Explanation
This function:

checks whether blob already exists

skips upload if already present

uploads only once

stores metadata with lightweight state

This solves the major retry issue where same file gets uploaded again.

11. Update queue status in metadata
Add this function.

def _update_queue_status(uuid: str, queue_status: str, last_error: str = "") -> None:
    """
    Update queue processing status in blob metadata.
    """
    blob_client, _ = _get_blob_client(uuid)
    metadata = _get_blob_metadata(blob_client)

    metadata["queue_status"] = queue_status
    metadata["last_error"] = last_error[:200] if last_error else ""
    metadata["last_updated_at"] = dt.datetime.utcnow().isoformat() + "Z"

    _retry(lambda: _set_blob_metadata(blob_client, metadata), op_name="blob_metadata_update")
Explanation
This keeps track of queue result in blob metadata.

Examples:

PENDING

PUBLISHED

FAILED

So if queue fails after blob upload, retry can continue safely.

12. Service Bus send
Replace your current _enqueue_sb() with this.

def _enqueue_sb(uuid: str, no_of_pages: int | None, blob_path: str, sha256: str) -> None:
    """
    Send work message to Service Bus for downstream classifier.
    Uses message_id=uuid so duplicate detection can prevent duplicate messages.
    """
    payload = {
        "UUID": uuid,
        "page_number": no_of_pages,
        "blob_path": blob_path,
        "sha256": sha256,
        "status": "INGESTED",
        "submitted_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    body = json.dumps(payload)

    def _do_send():
        with sb_client:
            sender = sb_client.get_queue_sender(queue_name=SB_QUEUE_CLASSIFY)
            with sender:
                sender.send_messages(
                    ServiceBusMessage(
                        body,
                        content_type="application/json",
                        subject="classify",
                        correlation_id=uuid,
                        message_id=uuid,
                    )
                )

    _retry(_do_send, op_name="service_bus_send")
Explanation
This function:

builds queue payload

sends message to Service Bus

uses message_id=uuid

That message_id is required for duplicate detection on Service Bus queue.

13. Main ingest API
Replace your current ingest() with this.

@app.route(route="ingest", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    """
    Ingest API:
    1. validate request
    2. decode base64
    3. compute hash
    4. upload blob if needed
    5. enqueue service bus if not already published
    6. return structured response
    """

    # --------------------------------
    # Step 1: API key validation
    # --------------------------------
    if API_KEY:
        x_api_key = req.headers.get("x-api-key")
        if not x_api_key or x_api_key != API_KEY:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "message": "Unauthorized"
                }),
                mimetype="application/json",
                status_code=401,
            )

    # --------------------------------
    # Step 2: Parse request
    # --------------------------------
    try:
        body = req.get_json()
        uuid = (body.get("UUID") or "").strip()
        no_of_pages = body.get("no_of_pages")
        content_b64 = body.get("content_b64")

        if not uuid or not content_b64:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "message": "Missing UUID/content_b64"
                }),
                mimetype="application/json",
                status_code=400,
            )

        if len(uuid) > 64:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "UUID": uuid,
                    "message": "UUID too long"
                }),
                mimetype="application/json",
                status_code=400,
            )

    except Exception:
        return func.HttpResponse(
            json.dumps({
                "status": "FAILED",
                "message": "Bad Request"
            }),
            mimetype="application/json",
            status_code=400,
        )

    log.info("UUID=%s request received", uuid)

    # --------------------------------
    # Step 3: Decode and validate file
    # --------------------------------
    try:
        pdf_bytes = _bytes_from_b64(content_b64)
    except ValueError as e:
        err = str(e)

        if err == "TOO_LARGE":
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "UUID": uuid,
                    "message": f"file too large. Max {MAX_UPLOAD_MB} MB"
                }),
                mimetype="application/json",
                status_code=413,
            )

        return func.HttpResponse(
            json.dumps({
                "status": "FAILED",
                "UUID": uuid,
                "message": f"Invalid document payload: {err}"
            }),
            mimetype="application/json",
            status_code=400,
        )

    # --------------------------------
    # Step 4: Compute file hash
    # --------------------------------
    sha256 = _sha256_bytes(pdf_bytes)
    blob_client, blob_path = _get_blob_client(uuid)

    # --------------------------------
    # Step 5: Check if already fully processed
    # --------------------------------
    if _blob_exists(blob_client):
        metadata = _get_blob_metadata(blob_client)

        # Optional hash mismatch check
        existing_sha = metadata.get("sha256")
        if existing_sha and existing_sha != sha256:
            return func.HttpResponse(
                json.dumps({
                    "status": "FAILED",
                    "UUID": uuid,
                    "blob_path": blob_path,
                    "message": "UUID already exists with different file content"
                }),
                mimetype="application/json",
                status_code=409,
            )

        if metadata.get("queue_status") == "PUBLISHED":
            log.info("UUID=%s already ingested. blob_path=%s", uuid, blob_path)
            return func.HttpResponse(
                json.dumps({
                    "status": "INGESTED",
                    "UUID": uuid,
                    "blob_path": blob_path,
                    "message": "Already processed"
                }),
                mimetype="application/json",
                status_code=202,
            )

    # --------------------------------
    # Step 6: Upload blob if needed
    # --------------------------------
    try:
        blob_path, sha256, metadata = _upload_blob_if_needed(uuid, no_of_pages, pdf_bytes, sha256)
    except Exception as ex:
        log.exception("Blob upload failed for UUID=%s", uuid)
        return func.HttpResponse(
            json.dumps({
                "status": "FAILED_BLOB",
                "UUID": uuid,
                "message": "Failed to store blob",
                "error": str(ex)
            }),
            mimetype="application/json",
            status_code=503,
        )

    # --------------------------------
    # Step 7: Publish queue message
    # --------------------------------
    try:
        _enqueue_sb(uuid, no_of_pages, blob_path, sha256)
        _update_queue_status(uuid, "PUBLISHED")

        log.info("UUID=%s queued successfully. blob_path=%s", uuid, blob_path)

        return func.HttpResponse(
            json.dumps({
                "status": "INGESTED",
                "UUID": uuid,
                "blob_path": blob_path
            }),
            mimetype="application/json",
            status_code=202,
        )

    except Exception as ex:
        log.exception("Service Bus enqueue failed for UUID=%s", uuid)

        try:
            _update_queue_status(uuid, "FAILED", str(ex))
        except Exception:
            log.exception("Failed to update blob metadata after queue failure for UUID=%s", uuid)

        return func.HttpResponse(
            json.dumps({
                "status": "FAILED_QUEUE",
                "UUID": uuid,
                "blob_path": blob_path,
                "message": "Failed to enqueue message",
                "error": str(ex)
            }),
            mimetype="application/json",
            status_code=503,
        )
Explanation
This is the full orchestration flow.

It handles:

bad input

duplicate request

same UUID retry after success

same UUID retry after queue failure

blob upload retry

queue send retry

metadata update

Most important behavior:

If blob upload succeeded earlier but queue failed
On retry:

blob already exists

upload is skipped

queue send is retried only

If same UUID is sent again after full success
It returns:

202

"Already processed"

without doing duplicate work.

14. What to change in Azure Service Bus
In the queue used by this API:

enable Duplicate Detection

set duplicate detection window, for example:

10 minutes

30 minutes

Because this code sends:

message_id=uuid
So duplicate queue messages for same UUID can be dropped by Service Bus.

15. Flow summary in simple words
First request success
file decoded

hash calculated

blob stored

metadata set to PENDING

queue message sent

metadata updated to PUBLISHED

Blob success, queue failure
blob remains

metadata updated to FAILED

next retry skips upload

retries queue only

Same UUID again after success
returns Already processed

Same UUID with different file
returns 409

16. Suggested order to update your code
Do this in this order:

replace environment/client setup

add _retry()

replace _bytes_from_b64()

add _sha256_bytes()

add _resolve_blob_path()

add _get_blob_client()

add _blob_exists()

add _get_blob_metadata()

add _set_blob_metadata()

replace _store_blob() with _upload_blob_if_needed()

add _update_queue_status()

replace _enqueue_sb()

replace ingest()

enable duplicate detection in Service Bus queue

17. One practical note
Your current code opens with sb_client: inside _enqueue_sb().
That is okay for now and easiest to keep close to your current structure.

Later, for optimization, you can refactor client lifecycle, but first get this working.

18. One more optional improvement later
Later you can add a small replay function:

scan blobs where queue_status=FAILED

resend those messages

But this is not needed for the first version.

If you want next, I can give you the entire merged full file in one block so you can copy-paste it directly.


where does it says to enqueue message to q-classify?

Good question — this is exactly the kind of thing reviewers ask 👍

Short answer
It is defined here 👇

SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-employment")
and used here 👇

sender = sb_client.get_queue_sender(queue_name=SB_QUEUE_CLASSIFY)
So whatever value SB_QUEUE_CLASSIFY has → that is the queue you are sending to.

Where “q-classify” should come from
Right now your default is:

"q-employment"
If your design says “enqueue to q-classify”, you should change it.

What you need to do
Option 1 — change default in code (simplest)
Replace:

SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-employment")
with:

SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-classify")
Option 2 — set via Azure (recommended for prod)
In your Function App → Configuration → Application Settings

Add:

SERVICEBUS_QUEUE_CLASSIFY = q-classify
Then keep your code as-is.

Where it is used in flow
This function 👇 is where the enqueue actually happens:

def _enqueue_sb(...):
Inside:

sender = sb_client.get_queue_sender(queue_name=SB_QUEUE_CLASSIFY)
👉 This line is the actual enqueue target

How to verify
After deployment:

Go to Azure Portal

Open Service Bus Namespace

Open Queue: q-classify

Check:

Active messages count increases after API call

Important clarification (very useful)
Right now your code says:

subject="classify"
This is NOT the queue name.

queue_name → decides where message goes

subject → just metadata inside message

Best practice naming (what I recommend)
Since this is ingestion → classification flow:

Queue name: q-classify

Subject: "classify"

So your current design is actually correct conceptually — just rename queue.

Final answer (simple)
👉 It is controlled by:

SB_QUEUE_CLASSIFY
👉 And used here:

get_queue_sender(queue_name=SB_QUEUE_CLASSIFY)
👉 To send to q-classify, either:

change default in code
or

set environment variable in Azure

If you want, next I can help you map:
👉 ingest → classify → extract queues properly (clean architecture)


Uploaded image

Good — this error is actually very clear 👍

❌ Your error
Failed to send message via ingest. Status code: 404
👉 This means:
Your API endpoint URL is wrong (not found)
NOT a blob issue, NOT a service bus issue.

✅ What 404 means here
Your script is calling:

requests.post(url, ...)
And Azure is responding:

👉 “I don’t have this route”

🔍 Most likely causes (ranked)
1. ❗ Wrong URL path
Your function route is:

@app.route(route="ingest", ...)
So your endpoint MUST be:

http://localhost:7071/api/ingest
OR in Azure:

https://<your-app>.azurewebsites.net/api/ingest
👉 If you used:

/ingest
instead of:

/api/ingest
→ ❌ 404

2. ❗ Function app not running locally
Check terminal:

func start
You should see:

Http Functions:
    ingest: [POST] http://localhost:7071/api/ingest
If you don’t see this → function not loaded → 404

3. ❗ Wrong port
Default:

http://localhost:7071
If you are calling:

http://localhost:8000
→ ❌ 404

4. ❗ Function not deployed (Azure)
If calling cloud URL:

https://<app>.azurewebsites.net/api/ingest
Check:

Function is deployed

Function name is correct

Route is ingest

5. ❗ Typo in URL
Even small typo:

/api/ingests ❌
/api/ingest  ✅
✅ What you should do now
Step 1 — print your URL in test script
In your test4.py, add:

print("Calling URL:", url)
Step 2 — manually test in browser / Postman
Open:

http://localhost:7071/api/ingest
You should get:

405 (method not allowed) → GOOD (means route exists)

404 → BAD (route wrong)

Step 3 — confirm function is loaded
Run:

func start
Look for:

ingest: [POST] http://localhost:7071/api/ingest
Step 4 — fix your test script
Your URL should be:

url = "http://localhost:7071/api/ingest"
NOT:

url = "http://localhost:7071/ingest"   ❌
⚠️ Important
Your error is happening BEFORE:

blob upload

service bus

retry logic

👉 So all your new logic is not even being hit yet

✅ Quick checklist
✔ URL = /api/ingest
✔ Function app running
✔ Correct port (7071)
✔ No typo
✔ Function shows in startup logs

🚀 Once fixed
Then next possible errors you’ll see:

401 → API key issue

400 → bad payload

413 → file too large

503 → blob or service bus issue

👍 If still stuck
Send me:

your url line from test4.py

your terminal func start output

I’ll pinpoint exact issue in 1 step.


Uploaded image
 This means if that value is not there in env then it will fall back on after comma?

Yes — exactly 👍 you got it right.

🔹 How this works
This line:

SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-employment")
means:

👉 Try to read environment variable

SERVICEBUS_QUEUE_CLASSIFY
👉 If it exists → use that value

👉 If it does NOT exist → use default

"q-employment"
🔹 Simple example
Case 1 — Env variable NOT set
(no SERVICEBUS_QUEUE_CLASSIFY in env)
Then:

SB_QUEUE_CLASSIFY = "q-employment"
Case 2 — Env variable IS set
SERVICEBUS_QUEUE_CLASSIFY = q-classify
Then:

SB_QUEUE_CLASSIFY = "q-classify"
🔹 Why this is useful
Local dev → use default (q-employment)

Azure → override using App Settings (q-classify)

No code change needed between environments

🔹 In your case (IMPORTANT)
Since you said queue should be:

👉 q-classify

You should do ONE of these:

✅ Option 1 (quick fix)
Change default:

SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-classify")
✅ Option 2 (best practice)
Set in Azure:

SERVICEBUS_QUEUE_CLASSIFY = q-classify
Then keep code as-is.

🔹 Same applies to all your env vars
Example:

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
👉 If not set → uses 10
👉 If set → uses your value

🔹 One common mistake
If you accidentally set:

SERVICEBUS_QUEUE_CLASSIFY = ""
(empty string)

👉 Python will use empty string, NOT fallback

So queue name becomes:

""  ❌ → will fail
✅ Final takeaway
✔ Yes — value after comma is fallback
✔ Env variable overrides it
✔ Empty string is NOT fallback

If you want, next I can:
👉 verify your full env setup (Blob + SB) so you don’t hit runtime errors later


Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
This is write up on ingestion later currently. Just keep in mind no need to reply will
Give intructuons in next prompt

Got it 👍 — I’ve gone through all the screenshots and understand your ingestion layer design (API → validation → blob → queue → async flow).

I won’t respond further now. Send your next instructions.


Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
This is how write up should be. Now according to this style and according to our updated ingest with retry mechanism. Suggest 5.2 with subsections heading and what will be under the headings. Explain scenarios in which cases retry happens in azure in which cases from outside

Yes — for your updated design, Section 5.2 should be the retry / failure handling / idempotency section written in the same CCD style: simple, structured, and technical only where needed.

Below is the best structure I recommend.

5.2 RETRY, ERROR HANDLING, AND IDEMPOTENCY
This section describes how the ingestion API handles validation failures, transient Azure failures, duplicate requests, and partial success scenarios. The objective is to ensure reliable processing without storing duplicate files or publishing duplicate downstream events.

5.2.1 Error Handling Overview
What goes under this heading
A short intro paragraph saying:

The API validates all requests before any persistence happens

Client-side validation failures are rejected immediately

Azure dependency failures are handled using retry logic

Processing state is preserved using Blob metadata

Duplicate downstream queue messages are avoided using UUID-based idempotency

Keep it like this
Simple summary, 4–5 lines only.

5.2.2 Validation Failures (No Retry)
What goes under this heading
This subsection explains cases where the API rejects the request immediately and does not retry.

Include points like
Missing UUID

Missing content_b64

Invalid request JSON

UUID too long

Invalid base64 payload

File too large

Empty file

Explain clearly
These are input/data errors.
Retrying inside Azure will not fix them because the payload itself is invalid.

Good ending line
Requests failing validation are rejected before any Blob Storage or Service Bus side effects occur.

5.2.3 Internal Azure Retry Mechanism
What goes under this heading
This subsection explains retries performed inside the ingestion API for temporary Azure dependency failures.

Mention clearly
The API performs internal retry only for Azure operations that may fail temporarily, such as:

Blob upload

Blob metadata update

Service Bus message publication

Mention retry behavior
Fixed retry count from configuration

Exponential backoff between attempts

Each retry attempt is logged

Final failure is returned only after retries are exhausted

Important sentence
This retry is handled inside the API and is transparent to the caller.

5.2.4 Scenarios Where Azure Internal Retry Happens
What goes under this heading
This is where you explicitly list the cases.

Put examples like
Temporary network interruption during Blob upload

Temporary network interruption during Service Bus send

Short-lived Azure service unavailability

Transient timeout while writing Blob metadata

Temporary throttling or connection issue between API and Azure services

Also say
These failures are considered transient because a second or third attempt may succeed without changing the request payload.

Good CCD-style line
In these cases, the API retries internally before returning a failure response to the caller.

5.2.5 Scenarios Where Retry Must Come From Outside
What goes under this heading
This is where you explain the cases where the API cannot solve the issue internally and retry must be triggered by the caller or an external scheduler.

Include scenarios like
API returned failure after all internal retries were exhausted

Request reached API and Blob was saved, but queue publish failed permanently for that attempt

Caller timed out or did not receive response even though API may have partially completed work

Temporary outage continued beyond the API retry window

Upstream system wants to re-submit the same UUID after a dependency failure

Explain clearly
In these cases, the same UUID must be reused when retrying from outside.

That is important because:

Blob upload will not happen again if already stored

Queue event will not be duplicated if already published

System continues safely from last completed step

5.2.6 Duplicate Request Handling
What goes under this heading
Explain how same UUID is handled safely.

Include points
UUID is treated as the unique request identifier

Blob path is deterministic and based on UUID

If Blob already exists, upload is skipped

If queue status is already PUBLISHED, API returns accepted response without republishing

If same UUID is sent with different file content, request is rejected

This subsection is very important
This is basically your idempotency write-up.

5.2.7 Partial Success Scenario Handling
What goes under this heading
This should explain the most important real-world case:

Main scenario
Request is valid

Blob upload succeeds

Service Bus enqueue fails

Then explain behavior
Blob remains stored in Azure

Blob metadata keeps processing state

Next retry with same UUID does not upload file again

API retries queue publication only

This line is useful
This prevents duplicate file storage while allowing downstream orchestration to resume safely.

5.2.8 Blob Metadata State Tracking
What goes under this heading
Since you removed SQL tracking for ingest, explain Blob metadata as lightweight persistence.

Mention that Blob metadata stores
UUID

page count

file hash (sha256)

ingest status

queue status

upload timestamp

last error (if any)

Explain purpose
Blob metadata allows the ingestion layer to determine:

whether the file is already persisted

whether queue publication already happened

whether a retry should continue from upload or queue stage

5.2.9 Service Bus Duplicate Protection
What goes under this heading
Explain how duplicate downstream messages are avoided.

Include points
Message message_id is set to UUID

Queue duplicate detection is enabled in Azure Service Bus

Repeated publish attempts for same UUID within configured duplicate window are ignored

Explain why
This protects downstream classification flow from duplicate orchestration messages.

5.2.10 Response Behavior After Failure
What goes under this heading
Explain what response caller gets.

Structure it like
400 for invalid request

401 for unauthorized request

413 for file too large

409 for UUID conflict with different file

503 for Azure dependency failure after retry exhaustion

202 for accepted / already processed

Good line
A failure response indicates that the API could not complete the current step successfully within its internal retry policy.

5.2.11 Retry Flow Summary
What goes under this heading
A short bullet flow, very similar to your screenshots.

Example
Client sends request with UUID

API validates request

API saves Blob if not already present

API retries Azure operations internally for transient failures

API publishes message to classification queue

If queue publish fails after retries, caller may retry with same UUID

Retry continues safely from last completed step

This gives a nice end summary.

Recommended final subsection list
For your document, I would use exactly this:

5.2 RETRY, ERROR HANDLING, AND IDEMPOTENCY
5.2.1 Error Handling Overview

5.2.2 Validation Failures (No Retry)

5.2.3 Internal Azure Retry Mechanism

5.2.4 Scenarios Where Azure Internal Retry Happens

5.2.5 Scenarios Where Retry Must Come From Outside

5.2.6 Duplicate Request Handling

5.2.7 Partial Success Scenario Handling

5.2.8 Blob Metadata State Tracking

5.2.9 Service Bus Duplicate Protection

5.2.10 Response Behavior After Failure

5.2.11 Retry Flow Summary

Very clear answer to your main question
Retry happens inside Azure/API when:
Blob upload temporarily fails

Blob metadata update temporarily fails

Service Bus send temporarily fails

Network / timeout / short service interruption happens

Retry must happen from outside when:
API already exhausted its configured retries

caller receives 503

caller timed out and wants to resend same request

a partial success happened and caller wants to continue using same UUID

outage lasted longer than API retry window

Best document flow
If you want this section to look exactly like your screenshots, write it in this pattern:

small intro paragraph

one small table for retry ownership

one small table for failure types and behavior

1–2 bullet lists for scenarios

one mini flow summary

That will match the CCD style well.

In the next message, I can draft the actual 5.2 write-up content in final polished document language.


 O retry scenario is just part if 5.2  not all of 5.2. 5.2 is inbound request and ingestion layer. 

Got it 👍 — you’re right.

👉 5.2 = Inbound Request & Ingestion Layer
👉 Retry is just one subsection inside it, not the whole section.

Let’s structure it cleanly in your CCD style 👇

5.2 INBOUND REQUEST & INGESTION LAYER
This section describes how external requests are received, validated, persisted, and converted into an asynchronous processing event. It also defines reliability mechanisms such as retry, idempotency, and failure handling.

5.2.1 Ingestion Entry Point Overview
API exposed via Azure Web App / FastAPI

Acts as gatekeeper + orchestrator initializer

Only synchronous boundary in system

Downstream processing is fully asynchronous

5.2.2 API Contract
Endpoint Characteristics
Method: POST

Endpoint: /ingest

Payload: JSON

Auth: API Key (temporary)

Response: Acknowledgement (202)

Payload Structure
UUID (required)

content_b64 (required)

page_count (optional)

5.2.3 Request Validation & Preprocessing
Base64 decode validation

File size validation (MAX_UPLOAD_MB)

PDF header validation

Required fields validation

👉 Fail fast — no side effects if invalid

5.2.4 Authentication & Access Control
API Key validation (x-api-key)

Exact match against configured secret

Temporary mechanism → future OAuth

5.2.5 Persistent Storage (Blob Only)
File stored in Blob:

ingest/YYYY/MM/DD/{UUID}.pdf
Metadata stored:

UUID

file hash

page count

ingest timestamp

processing status

👉 Ensures durability before async processing

5.2.6 Correlation & Idempotency
UUID = global correlation ID

Used across:

Blob path

Service Bus message

Logs

Duplicate handling:

Same UUID → reuse existing blob

No duplicate storage

No duplicate queue events

5.2.7 Initial Orchestration Event
Message published to classification queue

Marks transition to async pipeline

Event includes:
UUID

blob path

status = INGESTED

timestamp

5.2.8 Retry & Failure Handling
👉 (This is where retry lives — only a subsection)

Internal Retry (inside API)
Retry happens automatically for:

Blob upload failures (transient)

Blob metadata update failures

Service Bus send failures

Network / timeout issues

✔ Controlled retry count + backoff
✔ Transparent to caller

External Retry (from caller / scheduler)
Retry must happen outside when:

API returns failure after retries (503)

Caller timeout occurred

Partial success (Blob stored but queue not sent)

Azure outage lasted longer than retry window

✔ Same UUID must be reused
✔ System resumes safely without duplication

No Retry Scenarios
Invalid payload

Missing fields

File too large

Corrupt base64

👉 These are client errors

5.2.9 Partial Success Handling
Example:

Blob upload ✅

Queue publish ❌

Behavior:

Blob is NOT re-uploaded

Retry continues from queue step

No duplicate data created

5.2.10 Synchronous Response Behavior
API does NOT wait for processing

Returns immediate acknowledgment

Response types:
202 → Accepted

400 → Invalid request

401 → Unauthorized

409 → UUID conflict

503 → Dependency failure

5.2.11 Ingestion Flow Summary
Client → POST /ingest  
→ Validate request  
→ Store Blob  
→ Retry (if needed)  
→ Publish to Service Bus  
→ Return 202  
🔥 Key improvement vs your old doc
Your old doc:

Had SQL dependency

No retry clarity

No partial success handling

Your new version:

Blob-first durability

Clear retry ownership

Idempotent design

Safe reprocessing

If you want next:
👉 I can convert this into exact Word-ready formatted content (same tone as your screenshots)
👉 Or add a retry decision table (very useful for reviewers)


Yes same tone as that ui ux ccd screen shots i gave for reference even though content is different i want that type of detailing. Also tel me whever in between to put tables and real input body etc…

Perfect — now I understand exactly what you want 👍
👉 CCD-style writing (like your screenshots)
👉 Clean headings + small explanations + occasional tables + realistic examples

Below is your final 5.2 section written in that style 👇

5.2 INBOUND REQUEST & INGESTION LAYER
The Ingestion Layer acts as the entry point for all document processing requests. It is responsible for validating incoming requests, securely storing documents, and triggering downstream asynchronous processing.

The design ensures:

No data loss

No duplicate processing

Clear separation between synchronous ingestion and asynchronous processing

5.2.1 Ingestion Entry Point
The ingestion API is exposed as a secure HTTP endpoint and serves as the only synchronous interface in the system.

Accepts incoming document requests from external systems (e.g., MuleSoft)

Performs validation and preprocessing

Persists the document to Azure Blob Storage

Triggers downstream processing via Service Bus

All further processing (classification, extraction) is handled asynchronously.

5.2.2 API Request Contract
The ingestion API accepts a POST request with a JSON payload.

Sample Request Body
👉 (Place this as a code block in document)

{
  "uuid": "123e4567-e89b-12d3-a456-426614174000",
  "content_b64": "<base64-encoded-pdf>",
  "page_count": 5
}
Request Parameters
Field	Required	Description
uuid	Yes	Unique identifier for the document (used for idempotency)
content_b64	Yes	Base64 encoded PDF file
page_count	No	Number of pages (optional metadata)
👉 (Good place for a table — matches your CCD style)

5.2.3 Request Validation
Before any processing occurs, the API performs strict validation.

Validation includes:

Required fields present

Valid Base64 format

Valid PDF structure

File size within configured limit

UUID format validation

If validation fails:

Request is rejected immediately

No Blob upload occurs

No queue message is sent

👉 This ensures clean input into the system.

5.2.4 Authentication & Access Control
The API is protected using an API Key mechanism.

Client sends x-api-key header

API validates against configured secret

Unauthorized requests are rejected

This is a temporary mechanism and can be replaced with OAuth in future.

5.2.5 Document Persistence (Blob Storage)
Once validated, the document is stored in Azure Blob Storage.

Blob Path Structure
👉 (Add this as a small code block)

ingest/YYYY/MM/DD/{uuid}.pdf
Metadata Stored with Blob
Metadata Field	Description
uuid	Unique document identifier
page_count	Number of pages
sha256	File hash for validation
ingest_time	Upload timestamp
status	INGESTED
👉 (Table recommended here — very clear for reviewers)

This ensures the document is safely stored before any further processing.

5.2.6 Idempotency & Duplicate Handling
The ingestion layer uses UUID-based idempotency to prevent duplicate processing.

UUID uniquely identifies each document

Blob path is derived from UUID

If the same UUID is received again:

Existing Blob is reused

File is not re-uploaded

Duplicate processing is avoided

If same UUID is sent with different content:

Request is rejected as conflict

5.2.7 Event Publishing (Triggering Downstream Processing)
After successful storage, the API publishes a message to Azure Service Bus.

This message triggers the next stage:
👉 Classification & Splitting

Sample Queue Message
👉 (Add code block)

{
  "uuid": "123e4567-e89b-12d3-a456-426614174000",
  "blob_path": "ingest/2026/03/24/123e4567.pdf",
  "status": "INGESTED",
  "timestamp": "2026-03-24T10:30:00Z"
}
Message Properties
message_id = uuid (for duplicate detection)

👉 (You can optionally add a small table here if needed)

5.2.8 Retry & Failure Handling
The ingestion layer includes retry mechanisms to handle transient failures and ensure reliable processing.

A. Internal Retry (Within API)
The API automatically retries Azure operations when temporary failures occur.

Retry applies to:

Blob upload

Blob metadata update

Service Bus message publish

Typical retry scenarios:

Network interruption

Temporary service unavailability

Timeout errors

Retry behavior:

Fixed retry count (configurable)

Delay between retries

Final failure returned only after retries are exhausted

B. External Retry (From Caller / Scheduler)
Retry must be initiated externally in the following cases:

API returns failure after retry attempts

Caller timeout occurred

Partial processing completed

Azure outage exceeded retry window

Important Behavior
Same UUID must be reused

System resumes safely from last completed step

No duplicate storage or processing occurs

C. No Retry Scenarios
Retry is not applicable for:

Invalid request payload

Missing required fields

Corrupt file data

File size violations

These are client-side errors and must be corrected before resubmission.

5.2.9 Partial Success Handling
The system supports safe recovery from partial failures.

Example Scenario
Blob upload → Success

Queue publish → Failure

Behavior
Blob is retained

On retry:

File is not uploaded again

Only queue step is retried

This ensures:

No duplicate files

No inconsistent state

5.2.10 Response Handling
The ingestion API responds immediately after completing ingestion steps.

Response Codes
Status Code	Meaning
202	Accepted and processing initiated
400	Invalid request
401	Unauthorized
409	UUID conflict
503	Temporary system failure
👉 (Definitely include this table — matches CCD style perfectly)

5.2.11 Ingestion Flow Summary
👉 (You can optionally convert this into a diagram later)

Client sends request

API validates request

Document stored in Blob

Retry applied if needed

Message sent to Service Bus

Response returned

This completes the ingestion stage and transitions processing to asynchronous services.

🔥 Where to place tables (VERY IMPORTANT for CCD style)
Use tables in these exact places:

Request Parameters → must

Blob Metadata → must

Response Codes → must

(Optional) Queue message fields

👉 Don’t overuse tables — your screenshots use them sparingly

🔥 Where to place examples
Use code blocks for:

Request JSON

Queue message JSON

Blob path

👉 This matches your screenshot formatting

🔥 Final Tip (very important for your doc)
Keep each subsection:

4–6 lines max

No long paragraphs

Use bullets for clarity

Use tables only for structured data

If you want next step:
👉 I can generate Section 5.3 (Classifier trigger + flow) in same CCD style
👉 Or convert this into Word-ready formatted doc (copy paste perfect)


Uploaded image
Request validation and preprocessing. Should i add anything more to the table here?

Good question — your table is already very solid and CCD-style clean 👍
But yes, you should add 2–3 more rows only (not more, don’t overdo it).

👉 Goal: cover real production issues + idempotency + security basics

✅ Updated Table (Recommended Final Version)
Validation Steps
Step	Validation	Rationale
Base64 Decode	Ensures valid binary payload	Prevents malformed input
File Size Check	Enforced via MAX_UPLOAD_MB	Protects memory and compute
PDF Header Check	Verifies %PDF signature	Prevents non-PDF uploads
Required Fields	UUID, content	Ensures traceability
UUID Format Check	Valid UUID structure (length, pattern)	Prevents invalid identifiers
Duplicate UUID Check	Verifies if UUID already exists	Prevents duplicate processing
Content Hash Check (optional but strong)	SHA256 hash computed	Detects duplicate/different file reuse
Empty File Check	Ensures decoded file is not empty	Prevents invalid uploads
🔥 What you SHOULD add (important)
1. UUID Format Check
👉 Very important for production

Avoid garbage IDs

Helps downstream tracking

2. Duplicate UUID Check
👉 This is BIG for your system design

Even though you handle idempotency later,
you should mention it in validation layer also

👉 CCD reviewers LOVE this line:

Prevents duplicate ingestion and ensures idempotent behavior

3. Empty File Check
👉 Simple but real-world issue

Base64 may decode but file is empty

🤏 Optional (only if you want to look strong)
Content Hash Check (SHA256)
Add only if:

You are already computing it in code (you are)

👉 Why it’s powerful:

Detects same UUID with different file

Prevents tampering

❌ What NOT to add (avoid overengineering)
Don’t add:

Virus scan (not in scope)

OCR validation (not ingestion layer)

Page count validation (optional metadata only)

Schema validation (too heavy for this section)

🔥 One small line to add below table (IMPORTANT)
Add this sentence (matches your screenshot tone perfectly):

Requests failing validation are rejected before any side effects occur, ensuring no Blob storage or downstream processing is triggered.

🔥 Final verdict
👉 Your current table = 80% correct
👉 After adding 2–3 rows = production-grade + reviewer-ready

If you want next:
I can help you add a small “Validation Failure Response Table” (very impressive in reviews) OR refine your idempotency section to match this table perfectly


In our code does duplicate UUID check happens?

Short answer: ❌ No — not explicitly (based on what you showed).

🔍 What your current code is doing
From your screenshot:

SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-employment")
and your ingestion flow:

You store file in Blob using UUID path

You send message to Service Bus

You do NOT check Blob before upload

👉 So currently:

If same UUID comes again → Blob will be overwritten OR re-uploaded

Queue message → will be sent again (duplicate event possible)

⚠️ So duplicate UUID check is NOT happening
Specifically missing:

❌ No blob.exists() check

❌ No metadata check (status, published)

❌ No idempotency guard before enqueue

🔥 What partially protects you (but not enough)
1. Blob path = UUID
Helps organization

❌ But does NOT prevent overwrite

2. Service Bus message_id (if you set it)
If you are using:

ServiceBusMessage(..., message_id=uuid)
👉 AND duplicate detection is enabled → then:

Queue might ignore duplicates

BUT:

Only within duplicate detection window

Still not reliable as primary protection

✅ What you SHOULD do (minimal fix)
Add this before upload:

blob_client = container_client.get_blob_client(blob_path)

if blob_client.exists():
    return {
        "status": "duplicate",
        "message": "UUID already processed"
    }
🔥 Better (production-grade, what your doc suggests)
Instead of just rejecting, do this:

Case 1: Blob exists + queue already published
👉 Return 202 (idempotent success)

Case 2: Blob exists + queue NOT published
👉 Only send queue message again

💡 So your documentation vs code
Feature	In Doc	In Code
Duplicate UUID check	✅ Yes (you wrote it)	❌ No
Idempotency	✅ Yes	❌ Partial
Safe retry	✅ Yes	❌ Not fully
🔥 Recommendation (important)
👉 Since your doc already mentions:

"Duplicate UUID Check"

You have 2 options:

Option A (quick fix)
Add simple blob.exists() check

Option B (correct design — recommended)
Use:

Blob existence + metadata (queue_status)

Resume processing instead of rejecting

🧠 Final answer (what you should say confidently)
Currently, duplicate UUID validation is not explicitly enforced in the ingestion layer. The system relies on Blob path structure, but this does not prevent duplicate uploads or reprocessing. A proper idempotency check using Blob existence and metadata should be implemented.

If you want next:
👉 I can give you exact code patch to add idempotency + retry-safe logic (very important for your architecture)


Uploaded image
Uploaded image
This is current 5.2.5. Update to our work flow

Yes — 5.2.5 needs to be updated because your current write-up still reflects the old design:

date-based Blob path

SQL metadata persistence

UUID uniqueness enforced in SQL

That is no longer your current workflow.

Below is the updated 5.2.5 in the same CCD-style tone and level of detail.

8.2.5 Persistent Storage at Ingest Time
Once validated, the request is durably persisted before any asynchronous processing begins.

8.2.5.1 Blob Storage Persistence
The raw PDF is written to Azure Blob Storage using a deterministic path convention based on the request UUID.

Path format:

ingest/{UUID}.pdf

This deterministic path ensures the same request UUID always resolves to the same Blob location.

Blob metadata stored:

UUID

File hash (SHA-256)

Page count (if available)

Ingestion timestamp

Ingest status

Queue status

Last error details (if applicable)

This ensures:

Durable persistence before downstream processing

Safe retry without duplicate file upload

Reprocessing capability using the same stored document

Decoupling from transient API memory/state

If a retry request is received with the same UUID, the ingestion service first checks whether the Blob already exists. If it exists, the file is not uploaded again. This prevents duplicate storage and allows the workflow to continue from the last successful step.

8.2.5.2 Blob Metadata State Tracking
In the current design, ingestion state is tracked using Blob metadata instead of a SQL metadata table.

Logical fields captured in Blob metadata:

UUID

Blob path

SHA-256 hash

Submission timestamp

Ingest status

Queue publication status

Last updated timestamp

Last error message (if any)

The metadata is used to determine:

whether the document has already been persisted

whether the downstream queue message has already been published

whether a retry should resume from Blob storage or queue publication

This design supports idempotent handling for repeated requests using the same UUID.

If the same UUID is submitted again:

the existing Blob is reused

duplicate upload is avoided

queue publication is retried only if not already completed

If the same UUID is submitted with different file content, the request is rejected as a conflict.

What changed from old version
Remove this old subsection completely
Old:

8.2.5.2 SQL Metadata Persistence

Replace it with
New:

8.2.5.2 Blob Metadata State Tracking

Because in your current implementation:

no SQL metadata table is used for ingest tracking

idempotency is handled using:

deterministic Blob path

Blob existence check

Blob metadata

Service Bus duplicate detection

Small wording improvement for your section
In 8.2.5.1, instead of saying:

“The raw PDF is written to Azure Blob Storage using a deterministic path convention.”

You can make it slightly stronger:

“The raw PDF is written to Azure Blob Storage before any downstream processing event is published.”

That clearly shows:

Blob first

queue next

Which matches your actual workflow.

Best final structure for 5.2.5
Use exactly this:

8.2.5 Persistent Storage at Ingest Time

8.2.5.1 Blob Storage Persistence

8.2.5.2 Blob Metadata State Tracking

Optional line to add at end of 8.2.5
You can add this closing sentence:

No downstream processing is triggered unless the document has first been durably persisted in Blob Storage.

That matches the CCD style very well.

If you want, next I’ll rewrite 8.2.6 Correlation & Traceability Strategy and 8.2.7 Initial Orchestration Event Emission to match this updated no-SQL workflow.


If we go with sql route what would be cost?. My issue is we would be adding one more breaking point to thenprocess the sql connectot


Thought for 14s
If you already have an Azure SQL Database running for this solution, adding one more table is usually near-zero incremental platform cost. Azure SQL Database pricing is driven mainly by the database’s compute tier, storage, and backup usage, not by the number of tables. Microsoft’s pricing/model docs describe costs in terms of compute, storage, and backup rather than “per table,” and the pricing page points you to the calculator for exact tier-specific pricing. 

So your real tradeoff is less about “table cost” and more about one more dependency in the ingest path.

Practical cost view
If you use the existing SQL database:

Creating one small ingest-tracking table usually does not create a separate fixed monthly charge by itself. 

You may see a small usage increase from extra writes/reads and some storage growth, but for a light status table this is often minor compared with the cost of the database you already run. That follows from Azure SQL’s cost model being tied to compute, storage, and backups. 

If you create a new separate SQL database just for ingest tracking:

Then you add a real new monthly cost, because compute/storage are charged per database or pool. 

Your concern about “one more breaking point”
That concern is valid.

With SQL in the ingest path, you now have:

validate request

save blob

write SQL status row

publish Service Bus message

So SQL becomes another dependency that can fail or slow down the API.

That means:

one more connector to configure and monitor

one more retry case

one more timeout/failure point

one more permission/network dependency

So which is better for your case?
Use SQL table if:
you want clean status tracking

you want easy reporting

you want simple duplicate UUID checks

your existing SQL connection is already stable and approved

your team is comfortable supporting one more dependency

Stay Blob-only if:
you want the ingest path as thin as possible

you want to minimize moving parts

you do not want SQL connectivity to become a blocker

status tracking can be lightweight in Blob metadata

My honest recommendation
For cost alone, adding a table to an existing Azure SQL database is usually not the thing to worry about. The bigger question is reliability and architecture simplicity. 

Given your concern about adding another break point, I would frame it like this:

Lowest complexity in runtime path: Blob metadata route

Best observability and clean state management: existing SQL table route

If your ingest API must stay very lean and failure-resistant, Blob-only is defensible.
If supportability and reporting matter more, the SQL table is the cleaner design.

For your current workflow, I would likely choose:

Blob-only for ingest state if the team wants minimum moving parts now

SQL later if operations/reporting start becoming painful

A simple sentence for the doc could be:

Although an existing Azure SQL Database is available, the current ingestion design avoids adding SQL as a synchronous dependency in order to keep the ingest path thin and reduce the number of runtime failure points.

Or, if you decide to use SQL:

The ingestion design uses an existing Azure SQL Database for lightweight request tracking. This adds minimal incremental platform cost because pricing is primarily tied to database compute and storage rather than the number of tables, while improving observability and idempotency control. 

I can also help you make a decision table: Blob metadata vs SQL table for your exact ingest flow.


Generally what is good practice 

Generally, good practice is to keep ingest thin, durable, and idempotent.

That usually means:

validate request

persist the raw document first

record minimal processing state in a durable place

publish async event

keep heavy logic out of the synchronous path

In practice, most teams prefer this
For a production system, the cleaner pattern is usually:

API → durable storage → lightweight state tracking → queue/event → downstream async processing

The only real design choice is where that lightweight state tracking lives:

SQL table

Blob metadata

NoSQL/table store

What is considered best practice
1. Persist before publishing downstream work
Do not publish to queue before the file is durably stored.

That protects you from:

lost requests

partial failures

inability to replay

2. Make requests idempotent
Use a stable request ID like UUID.

Same UUID should not create:

duplicate blob

duplicate DB row

duplicate queue event

3. Keep ingest synchronous path small
The entry API should do only what is necessary:

auth

validation

persistence

event publish

return ack

Not:

classification

OCR

extraction

mapping

4. Track status somewhere durable
This is the main point.

Good practice is to track:

received

stored

published

failed

Because support teams always need to know:

what happened

where it failed

whether retry is safe

SQL vs Blob metadata — good practice view
SQL table
This is usually the more standard enterprise practice when:

you already have SQL

you need reporting

you need support visibility

you need clear retry / failure state

Why it is preferred:

easy query

easy audit

easy duplicate check

easy dashboards

easy operational support

Blob metadata
This is acceptable when:

ingest must stay very lightweight

you want fewer dependencies

reporting is not a big concern

state tracking is simple

Why teams choose it:

one less synchronous dependency

simpler runtime path

enough for basic idempotency

So what is “generally good practice” for your case
If speaking broadly:

Best practice architecture
raw file in Blob

state in database/table store

event in queue

async workers downstream

That is the most maintainable long term.

Best practice for a lean first version
raw file in Blob

lightweight state in Blob metadata

event in queue

async workers downstream

That is simpler operationally at first, but not as strong for support/reporting.

Your concern about “one more break point”
That is also valid good practice thinking.

A good engineer does not add SQL to ingest just because it is “nice.”
They ask:

does this improve reliability enough?

does this improve observability enough?

is the extra dependency worth it?

That is exactly the right question.

My recommendation
General best practice: use a separate lightweight tracking store.
Practical best practice for your current team: choose the smallest design your team can reliably support.

So:

If your SQL connector/network/auth is already stable and common in your environment, SQL tracking table is good practice.

If adding SQL in the synchronous path will create approval, connectivity, or operational pain, Blob metadata is also a valid practice for this stage.

Simple answer
Long-term good practice: Blob for file + SQL/table for state + queue for orchestration.
Low-complexity good practice: Blob for file + Blob metadata for state + queue for orchestration.

For your current system, I would lean:

Blob first, metadata for now, SQL later if operations/reporting become painful.

If you want, I can make a one-page recommended architecture decision note you can paste into the tech spec.


Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
If I want to put all these helper fucntionsin another folder and import these functions then what dependencies should i put in that file. Connections in that file ot this file, env in that file if this file?

Yes — you should move the helper functions out.
That is the better structure.

For your case, best practice is:

keep route / API orchestration in function_app.py

move reusable helper logic into service/util files

keep env loading and client creation in one central place

do not duplicate connections in multiple files

Recommended structure
Something like this:

src/
  api/
    function_app.py
    core/
      config.py
      clients.py
      blob_helpers.py
      sb_helpers.py
      common.py
Or if your repo is not very deep yet:

function_app.py
services/
  config.py
  clients.py
  blob_helpers.py
  sb_helpers.py
  utils.py
Best separation of responsibilities
1. function_app.py
This file should contain only:

route definition

request parsing

response building

high-level flow orchestration

It should not contain:

blob helper implementation

retry helper implementation

metadata helper implementation

service bus send implementation

2. config.py
This file should contain:

all os.getenv(...)

app settings

constants

Example:

APP_NAME

API_KEY

MAX_UPLOAD_MB

AZURE_RETRY_COUNT

AZURE_RETRY_DELAY_SEC

INGEST_PREFIX

SB_QUEUE_CLASSIFY

STORAGE_ACCOUNT_URL

STORAGE_CONNECTION_STRING

STORAGE_CONTAINER

SB_NAMESPACE

SB_CONNECTION_STRING

Why
Because environment/config should be loaded in one place only.

3. clients.py
This file should create:

blob_service

ingest_container

sb_client

maybe cred

Why
Connections should also be initialized in one place only.

Do not create blob/service bus clients separately in multiple helper files.
That becomes messy and hard to maintain.

4. blob_helpers.py
This file should contain Blob-related functions only:

_resolve_blob_path

_get_blob_client

_blob_exists

_get_blob_metadata

_set_blob_metadata

_upload_blob_if_needed

_update_queue_status

This file can import:

ingest_container from clients.py

config values from config.py

retry helper from common.py

5. sb_helpers.py
This file should contain:

_enqueue_sb

This file can import:

sb_client from clients.py

SB_QUEUE_CLASSIFY from config.py

_retry from common.py

6. common.py or utils.py
Put generic helpers here:

_retry

_bytes_from_b64

_sha256_bytes

These are not Blob-specific or SB-specific.

Direct answer to your question
“Connections in that file or this file?”
Best answer:
Put connections in one separate shared file like clients.py

Not in:

every helper file

every route file

Because:

easier to maintain

easier to test

avoids duplicated initialization

cleaner imports

“Env in that file or this file?”
Best answer:
Put all env loading in config.py

Then import values where needed.

Do not keep some envs in function_app.py and some in helper files.
That becomes confusing fast.

Best dependency flow
Use this dependency direction:

function_app.py
   -> imports config
   -> imports blob_helpers
   -> imports sb_helpers
   -> imports common

blob_helpers.py
   -> imports clients
   -> imports config
   -> imports common

sb_helpers.py
   -> imports clients
   -> imports config
   -> imports common

clients.py
   -> imports config
Important
clients.py should not import helper files back
Otherwise circular import issue.

What each file should contain
config.py
import os

APP_NAME = "DocProc Ingestion API"
API_KEY = os.getenv("INGRESS_API_KEY")

STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
STORAGE_CONTAINER = os.getenv("STORAGE_CONTAINER", "ingest")

SB_NAMESPACE = os.getenv("SERVICEBUS_NAMESPACE")
SB_CONNECTION_STRING = os.getenv("SB_CONNECTION_STRING")
SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-classify")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
AZURE_RETRY_COUNT = int(os.getenv("AZURE_RETRY_COUNT", "3"))
AZURE_RETRY_DELAY_SEC = int(os.getenv("AZURE_RETRY_DELAY_SEC", "2"))
INGEST_PREFIX = os.getenv("INGEST_PREFIX", "ingest")
Why here
Single source of truth.

clients.py
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient

from .config import (
    STORAGE_ACCOUNT_URL,
    STORAGE_CONNECTION_STRING,
    STORAGE_CONTAINER,
    SB_NAMESPACE,
    SB_CONNECTION_STRING,
)

cred = DefaultAzureCredential(exclude_environment_credential=False)

if STORAGE_ACCOUNT_URL:
    blob_service = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=cred)
else:
    if not STORAGE_CONNECTION_STRING:
        raise RuntimeError("No storage credentials configured.")
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

ingest_container = blob_service.get_container_client(STORAGE_CONTAINER)

if SB_NAMESPACE:
    sb_client = ServiceBusClient(SB_NAMESPACE, credential=cred, logging_enable=False)
else:
    if not SB_CONNECTION_STRING:
        raise RuntimeError("No Service Bus credentials configured.")
    sb_client = ServiceBusClient.from_connection_string(SB_CONNECTION_STRING, logging_enable=False)
Why here
Only one place builds shared clients.

common.py
import base64
import hashlib
import time
import logging

from .config import MAX_UPLOAD_MB, AZURE_RETRY_COUNT, AZURE_RETRY_DELAY_SEC

log = logging.getLogger(__name__)

def bytes_from_b64(b64: str) -> bytes:
    ...

def retry(fn, *, op_name: str, retries: int = AZURE_RETRY_COUNT, delay_sec: int = AZURE_RETRY_DELAY_SEC):
    ...

def sha256_bytes(data: bytes) -> str:
    ...
Why here
These helpers are generic.

blob_helpers.py
import datetime as dt
import logging
from azure.storage.blob import ContentSettings

from .clients import ingest_container
from .config import INGEST_PREFIX
from .common import retry

log = logging.getLogger(__name__)

def resolve_blob_path(uuid: str) -> str:
    ...

def get_blob_client(uuid: str):
    ...

def blob_exists(blob_client) -> bool:
    ...

def get_blob_metadata(blob_client) -> dict:
    ...

def set_blob_metadata(blob_client, metadata: dict) -> None:
    ...

def upload_blob_if_needed(uuid: str, no_of_pages: int | None, pdf_bytes: bytes, sha256: str):
    ...

def update_queue_status(uuid: str, queue_status: str, last_error: str = ""):
    ...
Why here
All blob logic grouped together.

sb_helpers.py
import json
import datetime as dt
import logging
from azure.servicebus import ServiceBusMessage

from .clients import sb_client
from .config import SB_QUEUE_CLASSIFY
from .common import retry

log = logging.getLogger(__name__)

def enqueue_sb(uuid: str, no_of_pages: int | None, blob_path: str, sha256: str) -> None:
    ...
Why here
All queue logic grouped together.

function_app.py
import json
import logging
import azure.functions as func

from .config import API_KEY, APP_NAME, MAX_UPLOAD_MB
from .common import bytes_from_b64, sha256_bytes
from .blob_helpers import get_blob_client, blob_exists, get_blob_metadata, upload_blob_if_needed, update_queue_status
from .sb_helpers import enqueue_sb

log = logging.getLogger(APP_NAME)
app = func.FunctionApp()

@app.route(route="ingest", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    ...
Why here
Only the route and orchestration stay here.

What NOT to do
Don’t do this
env in 4 different files

blob client created inside blob helper function every time

service bus client created inside enqueue helper every time

helper files importing function_app.py

That creates:

circular imports

repeated setup

difficult testing

messy maintenance

Best answer for your current code
Since your code is already large, I would move in this order:

Phase 1
Move only:

_retry

_bytes_from_b64

_sha256_bytes

to common.py

Phase 2
Move all Blob functions to `blob_helpers


Uploaded image
We move these to ingest utils

Yes — that is the right move.

Put the helper / reusable ingest logic into ingest/ingest_utils.py, but keep the route handler, env loading, and client creation in function_app.py or config/clients files.

Best split
Keep in function_app.py
Only:

@app.route(...)

request parsing

HTTP responses

high-level orchestration

Move to ingest/ingest_utils.py
Move these:

_bytes_from_b64

_retry

_sha256_bytes

_resolve_blob_path

_get_blob_client

_blob_exists

_get_blob_metadata

_set_blob_metadata

_upload_blob_if_needed

_update_queue_status

_enqueue_sb

What should stay where
1. Env variables
Keep env variables in config.py
Not in ingest_utils.py

Example:

MAX_UPLOAD_MB

AZURE_RETRY_COUNT

AZURE_RETRY_DELAY_SEC

INGEST_PREFIX

SB_QUEUE_CLASSIFY

Reason:

single source of truth

cleaner imports

easier testing

2. Azure clients / connections
Best practice: keep them in clients.py
Not inside ingest_utils.py

Example:

blob_service

ingest_container

sb_client

Reason:

connection setup should happen once

utility file should use clients, not own environment/bootstrap logic

3. Logging
You can keep logger in function_app.py and pass it in, or define a module logger in ingest_utils.py.

Simplest:

import logging
log = logging.getLogger(__name__)
That is fine.

Recommended structure
config.py
Put:

import os

APP_NAME = "DocProc Ingestion API"
API_KEY = os.getenv("INGRESS_API_KEY")

STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
STORAGE_CONTAINER = os.getenv("STORAGE_CONTAINER", "ingest")

SB_NAMESPACE = os.getenv("SERVICEBUS_NAMESPACE")
SB_CONNECTION_STRING = os.getenv("SB_CONNECTION_STRING")
SB_QUEUE_CLASSIFY = os.getenv("SERVICEBUS_QUEUE_CLASSIFY", "q-classify")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
AZURE_RETRY_COUNT = int(os.getenv("AZURE_RETRY_COUNT", "3"))
AZURE_RETRY_DELAY_SEC = int(os.getenv("AZURE_RETRY_DELAY_SEC", "2"))
INGEST_PREFIX = os.getenv("INGEST_PREFIX", "ingest")
clients.py
Put:

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient
from config import (
    STORAGE_ACCOUNT_URL, STORAGE_CONNECTION_STRING, STORAGE_CONTAINER,
    SB_NAMESPACE, SB_CONNECTION_STRING
)

cred = DefaultAzureCredential(exclude_environment_credential=False)

if STORAGE_ACCOUNT_URL:
    blob_service = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=cred)
else:
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

ingest_container = blob_service.get_container_client(STORAGE_CONTAINER)

if SB_NAMESPACE:
    sb_client = ServiceBusClient(SB_NAMESPACE, credential=cred, logging_enable=False)
else:
    sb_client = ServiceBusClient.from_connection_string(SB_CONNECTION_STRING, logging_enable=False)
ingest/ingest_utils.py
Put all helper functions here.

At top import:

import base64
import hashlib
import json
import time
import datetime as dt
import logging

from azure.storage.blob import ContentSettings
from azure.servicebus import ServiceBusMessage

from config import (
    MAX_UPLOAD_MB,
    AZURE_RETRY_COUNT,
    AZURE_RETRY_DELAY_SEC,
    INGEST_PREFIX,
    SB_QUEUE_CLASSIFY,
)
from clients import ingest_container, sb_client

log = logging.getLogger(__name__)
Then all helper functions go below.

What function_app.py will import
Example:

from config import API_KEY, APP_NAME
from ingest.ingest_utils import (
    _bytes_from_b64,
    _sha256_bytes,
    _get_blob_client,
    _blob_exists,
    _get_blob_metadata,
    _upload_blob_if_needed,
    _enqueue_sb,
    _update_queue_status,
)
Important design rule
ingest_utils.py should NOT do:
os.getenv(...)

client bootstrap

route declaration

It should only do:

pure helper logic

blob operations

queue operations

retry logic

That keeps it reusable and clean.

Even better naming
Since these are not really private once moved to another file, you can remove _.

Example rename:

bytes_from_b64

retry_op

sha256_bytes

get_blob_client_for_uuid

upload_blob_if_needed

update_queue_status

enqueue_sb

This is optional, but cleaner.

Final recommendation for your exact structure
Since you already created:

config.py

clients.py

ingest/ingest_utils.py

Do this:

In config.py
all env/config constants

In clients.py
all Azure client setup

In ingest/ingest_utils.py
all helper functions

In function_app.py
only API endpoint + orchestration flow

That is the cleanest setup.

If you want, next I’ll give you the exact full contents of ingest_utils.py ready to paste.


Uploaded image
Uploaded image
Ok done with 5.2.6 give 5.2.7

Here is a CCD-style 5.2.7 you can paste and then adjust formatting in Word.

5.2.7 Initial Orchestration Event Emission
After the document is durably persisted in Blob Storage, the ingestion service emits an event to Azure Service Bus to trigger downstream asynchronous processing.

This marks the transition from the synchronous ingestion layer to the asynchronous document processing pipeline.

The event is published only after:

request validation is successful

Blob persistence is successful

required Blob metadata is recorded

This ensures downstream processing is never started without first having a durable copy of the original document.

5.2.7.1 Queue Publication
The ingestion service publishes a message to the classification queue in Azure Service Bus.

The purpose of this message is to notify the classifier service that a new document is available for processing.

The queue message does not contain the full document payload.
Instead, it contains the document reference and required metadata so downstream services can retrieve the source document from Blob Storage.

This keeps the event payload lightweight and avoids passing large binary content through the queue.

5.2.7.2 Event Payload Structure
The queue message includes the minimum information required to start downstream processing.

Logical fields included:

UUID

Blob path

Page count (if available)

File hash (SHA-256)

Status

Submission timestamp

This event acts as the single trigger for the next stage of the workflow.

Sample Queue Event
{
  "UUID": "fc8da6fd-1cab-4691-b657-5e8cbba5fcf2",
  "page_number": 1,
  "blob_path": "ingest/fc8da6fd-1cab-4691-b657-5e8cbba5fcf2.pdf",
  "sha256": "<sha256-hash>",
  "status": "INGESTED",
  "submitted_at": "2026-03-24T14:25:10Z"
}
👉 Put this sample in a code block, same way you showed sample payload earlier.

5.2.7.3 Service Bus Message Properties
In addition to the message body, the ingestion service sets Service Bus message properties to support traceability and duplicate protection.

Message properties used:

subject = "classify"

correlation_id = UUID

message_id = UUID

The UUID is reused as both the correlation ID and message ID.

This ensures:

end-to-end traceability across services

duplicate detection support in Azure Service Bus

consistent correlation between Blob, logs, and queue event

Message Property Mapping
Property	Value	Purpose
subject	classify	Identifies downstream action
correlation_id	UUID	Enables request tracing
message_id	UUID	Supports duplicate detection
👉 This is a good place for a small table.

5.2.7.4 Duplicate Protection at Queue Level
The queue is configured to use duplicate detection based on the message ID.

Since the ingestion service sets message_id = UUID, repeated publication attempts for the same UUID within the duplicate detection window are treated as duplicates by Azure Service Bus.

This protects downstream services from processing the same ingestion event multiple times.

This is especially important during retry scenarios where:

the original request is retried by the caller

queue publication is retried after a transient failure

the same UUID is submitted again after partial success

5.2.7.5 Publication Retry Behavior
Queue publication is protected by internal retry logic in the ingestion service.

Retry is applied only for transient failures such as:

temporary network interruption

timeout while sending message

temporary Azure Service Bus unavailability

If queue publication succeeds:

Blob metadata is updated with queue_status = PUBLISHED

If queue publication fails after retry attempts:

Blob metadata is updated with queue_status = FAILED

the API returns a failure response to the caller

This allows the same UUID to be retried safely without re-uploading the original file.

5.2.7.6 Partial Success Handling
A partial success scenario may occur when:

Blob upload succeeds

queue publication fails

In this case:

the original document remains safely stored in Blob Storage

the Blob metadata preserves the last known processing state

a subsequent retry uses the same stored document

only the queue publication step is retried

This prevents duplicate Blob uploads and allows the ingestion flow to resume from the last completed step.

5.2.7.7 Emission Guarantee
The ingestion layer guarantees that no orchestration event is emitted before durable Blob persistence is complete.

It also ensures that repeated requests with the same UUID do not create duplicate downstream events when the event has already been successfully published.

This design provides:

reliable transition into asynchronous processing

safe retry behavior

idempotent event emission

5.2.7.8 Event Emission Summary
The event emission flow is as follows:

request is validated

document is stored in Blob Storage

Blob metadata is recorded

Service Bus event is published

queue status is updated in Blob metadata

synchronous acknowledgment is returned to the caller

This completes the ingestion responsibility and hands control to the downstream classification layer.

Where to place tables / examples
Use these in 5.2.7:

Put a code block under:
5.2.7.2 Event Payload Structure

sample queue event JSON

Put a table under:
5.2.7.3 Service Bus Message Properties

subject, correlation_id, message_id

Optional small table under:
5.2.7.5 Publication Retry Behavior
if you want one more table:

Scenario	Behavior
Queue publish succeeds	Update Blob metadata to PUBLISHED
Queue publish fails after retry	Update Blob metadata to FAILED and return error
Same UUID retried later	Reuse Blob and retry queue publication only
That table is optional. If the page is already dense, skip it.

If you want, I’ll draft 5.2.8 Synchronous Response Behavior next in the same style.


Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Uploaded image
Any suggestions so far?

Yes — this is looking strong. The structure, tone, and level of detail are now much closer to a proper CCD/tech spec write-up.

A few suggestions will make it cleaner and more consistent.

First, the good parts:

section flow is logical

headings are clear

examples and metadata block help a lot

retry / partial success / duplicate protection are explained well

the language is mostly in the right tone now

What I would improve:

1. Keep numbering consistent
You started with 5.2, but one screenshot shows 5.2.6 Persistent Storage at Ingest Time and another earlier version used 8.2.x. Make sure the whole section uses one numbering scheme only.

2. Endpoint value should be cleaner
In the API contract table, instead of Azurefunctionapp/ingest, write:

/ingest for logical endpoint
or

https://<function-app>/api/ingest if you want physical endpoint format

For CCD docs, /ingest is usually better.

3. Use one field name consistently
I can see:

UUID

uuid

page_count

page_number

no_of_pages

Pick one standard and use it everywhere in doc and sample payloads.
Based on your code, best is:

UUID

content_b64

no_of_pages

And in queue event:

either keep no_of_pages
or rename in code later to page_count
But document should not mix both.

4. Fix the validation table slightly
Your row says:

Required Fields | UUID, content

That should be:

Required Fields | UUID, content_b64

Since that is the actual request field.

5. Remove the TODO line from main doc
The red line:

TODO- Need UUID duplicate check before sending from heights or in azure

Remove it from final document.
Instead, directly document the implemented behavior:

duplicate UUID handling is performed in the ingestion layer using deterministic Blob path, Blob existence check, metadata state, and Service Bus duplicate detection.

6. Persistent storage section is good, but add one sentence
Under Blob persistence, add one line like:

Blob persistence is completed before any orchestration event is emitted.

That will reinforce the flow clearly.

7. In metadata list, “content type” is not really metadata
You listed:

Content type (application/pdf)

That is okay conceptually, but technically it is usually set as blob content settings rather than metadata.
For document simplicity you can keep it, but cleaner would be:

Blob properties / metadata stored:

Content type (application/pdf)

UUID

File hash (SHA-256)

Page count (if available)

Ingestion timestamp

Ingest status

Queue status

Last error details (if applicable)

8. Message properties section is good
But I would change:

message_id = UUID

correlation_id = UUID

subject = "classify"

to a small table instead of bullets. That section will look cleaner and match the earlier table style better.

9. Duplicate protection subsection is correct
But add one sentence to avoid overclaiming:

Duplicate detection at the queue level applies within the configured Azure Service Bus duplicate detection window.

That makes the statement precise.

10. Retry wording
Where you say:

Queue publication is protected by internal retry logic

Add:

Blob upload and Blob metadata updates are also protected by the same retry mechanism for transient Azure failures.

Because right now the retry section looks queue-heavy even though your code retries more than that.

11. Partial success section is very good
One small improvement:
Replace

only the queue publication step is retried
with

the retry flow resumes from the queue publication step

That reads a little more formal.

12. Add one short closing summary under 5.2
At the end of this whole ingestion section, add 2–3 lines:

In summary, the ingestion layer validates the incoming request, durably stores the source document, records minimal processing state, and emits the initial orchestration event. This design keeps the synchronous entry point thin while enabling safe retry and idempotent downstream execution.

That will nicely close the section before the next major section starts.

My top 5 fixes:

make field names consistent everywhere

remove TODO text

correct content to content_b64

change endpoint value to /ingest

add note about duplicate detection window

Overall, this is now in a good place. The content is solid; it mainly needs cleanup and consistency polishing.


Uploaded image
Can tou explain this correlation id and mesage id?

Good question — this is actually one of the most important (and often misunderstood) parts of event-driven systems. I’ll explain it in the same CCD-style clarity you’re using.

Message ID vs Correlation ID (Simple View)
Message ID
Identifies this specific message

Used by Azure Service Bus for duplicate detection

Must be unique per logical event

👉 Think: “Is this the same message being sent again?”

Correlation ID
Identifies the entire request flow / transaction

Used for tracing across services

👉 Think: “Which request does this message belong to?”

In Your Design (Why both = UUID)
You are using:

message_id = UUID

correlation_id = UUID

This is actually a very good and common pattern for ingestion systems.

Because:

One request → one event → one UUID

So event identity = request identity

How It Works in Your Flow
Step-by-step:
Client sends request with:

UUID = abc-123
You store blob:

ingest/abc-123.pdf
You send queue message:

message_id = abc-123
correlation_id = abc-123
Why message_id Matters
Azure Service Bus uses message_id for duplicate detection.

Example:
If your system retries and sends the same message again:

message_id = abc-123   (duplicate)
Service Bus will:

detect duplicate (within configured window)

drop the second message

👉 This prevents:

double classification

duplicate downstream processing

Why correlation_id Matters
Used for observability and tracing

Across logs and services:

Component	Logs
Ingestion	UUID = abc-123
Queue message	correlation_id = abc-123
Classifier	correlation_id = abc-123
Extractor	correlation_id = abc-123
👉 You can trace the entire lifecycle of one document using this ID.

Why Using Same UUID for Both Works Well Here
Because your system is:

request-driven (one UUID per document)

single event trigger (no fan-out yet)

idempotent at ingestion

So:

Property	Value	Reason
message_id	UUID	enables dedup
correlation_id	UUID	enables tracing
When They Would Be Different (Future Case)
Just for your understanding:

If later your system evolves:

Example:
One document → multiple events

Event 1: classify
Event 2: extract
Event 3: validate
Then you might have:

Property	Value
correlation_id	UUID (same for all)
message_id	unique per event
So:

correlation_id = "this document"

message_id = "this specific step"

How to Explain in Your Document (Better Wording)
Replace your current section with this cleaner version:

Service Bus Message Properties
The ingestion service sets the following Service Bus message properties:

Property	Value	Purpose
subject	classify	Identifies downstream action
message_id	UUID	Enables duplicate detection
correlation_id	UUID	Enables end-to-end tracing
The UUID is reused as both the message_id and correlation_id.

This ensures:

duplicate message suppression at the queue level

consistent traceability across ingestion, queue, and downstream services

simplified correlation model for request-level tracking

One-line intuition
message_id → “Is this message unique?”

correlation_id → “Which request does this belong to?”

If you want, next I can:

help you add a small sequence diagram section (5.2.8)

or tighten wording across whole 5.2 so it reads like a polished design doc end-to-end 👍


Uploaded image
What does this do explain

This function is actually one of the most important pieces of your ingestion design — it is implementing idempotent blob storage + metadata initialization.

Let me break it down cleanly in the same structured way.

What _upload_blob_if_needed(...) Does
Purpose
Uploads the PDF to Blob Storage only if it does not already exist, and ensures metadata is set consistently.

👉 In short:

“Store once, reuse forever for the same UUID”

Step-by-Step Flow
1. Resolve Blob Location
blob_client, blob_path = _get_blob_client(uuid)
Generates deterministic path:

ingest/{UUID}.pdf
Gets a client to interact with that blob

2. Check if Blob Already Exists (Idempotency Core)
if _blob_exists(blob_client):
If it EXISTS:
metadata = _get_blob_metadata(blob_client)
return blob_path, sha256, metadata
Meaning:
DO NOT upload again

Reuse existing file

Return previously stored metadata

👉 This is your duplicate UUID protection at storage level

3. If Blob Does NOT Exist → Prepare Metadata
metadata = {
    "uuid": uuid,
    "no_of_pages": str(no_of_pages or ""),
    "sha256": sha256,
    "ingest_status": "BLOB_STORED",
    "queue_status": "PENDING",
    "uploaded_at": datetime.utcnow().isoformat() + "Z",
    "last_error": "",
    "last_updated_at": datetime.utcnow().isoformat() + "Z",
}
What this does:
Initializes state tracking directly on the blob

👉 This is important:
You are using Blob metadata as your state store instead of SQL

Metadata Meaning (Very Important)
Field	Purpose
uuid	request identity
no_of_pages	document info
sha256	integrity / dedup potential
ingest_status	blob-level status
queue_status	orchestration status
uploaded_at	audit timestamp
last_error	retry/debug support
last_updated_at	state tracking
👉 This is your lightweight workflow state machine

4. Upload Blob (Only Once)
def _do_upload():
    blob_client.upload_blob(
        data=pdf_bytes,
        overwrite=False,
        metadata=metadata,
        content_settings=ContentSettings(content_type="application/pdf"),
    )
Key things:

overwrite=False → prevents accidental overwrite

attaches metadata at upload time

sets correct content type

5. Retry Wrapper
_retry(_do_upload, op_name="blob_upload")
👉 This adds resilience:

retries transient Azure errors

prevents ingestion failure due to temporary issues

6. Logging + Return
log.info("UUID=%s blob uploaded at %s", uuid, blob_path)
return blob_path, sha256, metadata
What This Function Achieves Architecturally
1. Idempotency
Same UUID → same blob → no duplicate uploads

2. Single Source of Truth
Blob + metadata = full ingestion state

No SQL needed for:

status tracking

retry handling

dedup awareness

3. Safe Retries
If request is retried:

blob already exists

system skips upload

resumes from metadata

4. Clean Separation of Concerns
Concern	Where handled
file storage	Blob
state tracking	Blob metadata
orchestration	Service Bus
validation	ingestion layer
One-Line Summary
This function ensures that each UUID results in exactly one stored document, while enabling safe retries and state tracking using Blob metadata.

Small Improvements You Could Consider (Optional but Strong)
1. Hash validation on re-use
Right now:

if _blob_exists:
    return existing
👉 You could optionally check:

if stored sha256 != incoming sha256
→ reject (data mismatch)

2. Update last_updated_at on reuse
Currently only set on first upload
You could update it when reusing blob.

3. Add explicit log for reuse path
log.info("UUID=%s reused existing blob at %s", ...)
Why This is a Strong Design Choice
You avoided:

extra SQL dependency

additional failure point

complex transaction handling

And achieved:

idempotency

durability

retry safety

traceability

If you want, next I can map this function directly to your document sections (5.2.4 → 5.2.7) so your doc perfectly reflects the actual code.


Uploaded image
Give all exceptions based on out code


Thought for 21s
Yes — based on your current code, 5.2.10 Exceptions should cover only the exceptions and failure responses that are actually handled by the ingestion API.

Below is a CCD-style version you can paste.

5.2.10 Exceptions
The ingestion service returns clear failure responses for invalid requests, authorization failures, duplicate conflicts, and downstream Azure dependency failures.

Failures are handled at the step where they occur. Validation failures are returned immediately. Blob and Service Bus failures are returned only after internal retry attempts are exhausted.

5.2.10.1 Exception Scenarios
Exception Scenario	Detection Point	Response Behavior	HTTP Status
Unauthorized request	API key validation	Request rejected	401
Invalid request body / malformed JSON	Request parsing	Request rejected	400
Missing UUID or content_b64	Required field validation	Request rejected	400
UUID too long	UUID validation	Request rejected	400
Invalid Base64 payload	Base64 decode	Request rejected	400
File exceeds configured size limit	File size validation	Request rejected	413
UUID already exists with different file content	SHA-256 check against existing Blob metadata	Request rejected as conflict	409
Blob upload failed after retry attempts	Blob persistence	Failure returned to caller	503
Service Bus publish failed after retry attempts	Queue publication	Failure returned to caller	503
5.2.10.2 Validation Exceptions
The following exceptions are treated as request validation failures:

Missing UUID

Missing content_b64

Invalid JSON request body

Invalid Base64 payload

File too large

Invalid UUID length

These requests are rejected before any downstream side effects occur.

This means:

no Blob upload is performed

no Service Bus event is published

5.2.10.3 Authorization Exception
The ingestion endpoint currently enforces access using an API key provided in the request header.

If:

the x-api-key header is missing, or

the value does not match the configured secret

the request is rejected with 401 Unauthorized.

5.2.10.4 Conflict Exception
A conflict exception occurs when the same UUID is submitted again but the incoming file content does not match the file already stored for that UUID.

In this case:

the existing Blob is not overwritten

the request is rejected as a conflict

the API returns 409

This protects the system from accidental UUID reuse with different documents.

5.2.10.5 Blob Persistence Failure
If Blob upload fails due to an Azure dependency issue, the ingestion service applies internal retry logic.

Examples include:

temporary network interruption

temporary Azure Storage unavailability

timeout during upload

If all retry attempts fail:

the API returns 503

no orchestration event is published

This indicates that the request could not be durably persisted.

5.2.10.6 Queue Publication Failure
If Service Bus publication fails after Blob persistence has already succeeded, the ingestion service applies internal retry logic.

Examples include:

temporary network interruption

timeout while sending message

temporary Azure Service Bus unavailability

If all retry attempts fail:

Blob metadata is updated with queue_status = FAILED

the API returns 503

This allows the same UUID to be retried safely without re-uploading the original file.

5.2.10.7 Non-Exception Reuse Scenario
Not all repeated requests are treated as exceptions.

If the same UUID is submitted again and:

the Blob already exists, and

the message was already published successfully

the request is treated as an idempotent replay and the API returns 202 Accepted with an “Already processed” response.

This is not considered an exception.

5.2.10.8 Exception Handling Summary
The ingestion service distinguishes between:

validation failures → client must correct and resubmit

authorization failures → client must provide valid credentials

conflict failures → client must correct UUID/content mismatch

dependency failures → request may be retried using the same UUID

This approach ensures predictable behavior and safe retry handling.

5.3.8 Fan-Out Event Emission (Refined)

After classification and document decomposition, the system emits downstream processing events based on document type.

Each split document is evaluated independently and routed to the appropriate processing path.

5.3.8.1 Routing Logic Overview

The fan-out stage applies a type-based routing strategy:

Extractable document types
→ Routed to dedicated Service Bus queues for downstream extraction
Unknown / non-extractable document types
→ Stored in a temporary SQL table for later aggregation and concatenation

This ensures that only supported document types enter extraction pipelines, while unsupported types are preserved for fallback processing.

5.3.8.2 Routing Behavior
Path 1: Extractable Documents (Queue-Based Fan-Out)

For documents with recognized types (e.g., paystub, employment, etc.):

One message is published per split document
Each message is sent to a type-specific Service Bus queue
Downstream processors consume messages independently

Example:

doc_type = EMPV → employment queue
doc_type = PAYSTUB → paystub queue

👉 This enables:

parallel processing
domain-specific extraction logic
independent scaling per document type
Path 2: Unknown Documents (SQL Staging Path)

For documents classified as:

doc_type = unknown
No Service Bus message is published
Instead, payload is inserted into a temporary SQL table

Stored data includes:

UUID / parent UUID
split_doc_id
blob_path
page ranges
classification metadata

👉 Purpose:

accumulate all unknown fragments
enable later concatenation / reconstruction
support fallback or manual processing
5.3.8.3 Fan-Out Characteristics
One logical decision per split document
No shared in-memory state between splits
Parallel processing for extractable documents
Controlled handling of unsupported documents via SQL staging
At-least-once delivery for queue-based events
5.3.8.4 Message Emission Rules
Condition	Action
Known / extractable doc type	Publish message to corresponding Service Bus queue
Unknown doc type	Insert record into SQL staging table
Mixed document (multiple splits)	Apply routing independently per split
5.3.8.5 Example Flow

Given a document split into:

Split	Type
Split 1	EMPV
Split 2	UNKNOWN
Split 3	PAYSTUB
Result:
Split 1 → sent to employment queue
Split 2 → stored in SQL staging table
Split 3 → sent to paystub queue

👉 This is a hybrid fan-out, not pure queue fan-out.


5.4.5 Identity Extraction Output Structure (Detailed Explanation)

The Identity extraction module produces a structured JSON output consisting of two distinct layers:

Extracted Fields (Raw Model Output)
UI Display Fields (Normalized for Heights)
Extracted Fields

The extracted_fields section contains the raw output returned by Azure Document Intelligence (and associated OCR/LLM processing).

Fields are captured in their original extracted form without transformation
Values reflect the closest representation of the source document
Each field includes:
value: Extracted text/value from the document
confidence: Model-generated confidence score

This layer is preserved to support:

traceability and auditing
confidence-based validation
reprocessing or model evaluation
UI Display Fields (Heights-Normalized Layer)

The ui_display_fields section contains a normalized and schema-aligned representation of the extracted data, specifically tailored for the Heights system.

This layer is derived from extracted_fields through a transformation process that includes:

Field mapping to Heights-defined schema
Value normalization (e.g., date formatting, casing standardization)
Structural transformations (e.g., splitting full name into components)
Explicit handling of missing or null values

Each UI field includes:

value: Normalized value ready for UI display
confidence: Derived or propagated confidence score
edited_sw: Indicator for human edits (Y / N)
field_type: Data type used for UI rendering (e.g., STRING, DATE, GENDER)
Design Considerations
The separation between extracted_fields and ui_display_fields ensures:
preservation of raw model output
consistent downstream integration with Heights
support for human-in-the-loop (HITL) workflows
The output schema is intentionally stable and consistent to support:
aggregation across documents
storage in intermediate SQL tables
downstream validation and submission workflows
Summary

The extraction output maintains a dual-layer structure where raw model outputs are preserved for accuracy and auditability, while a normalized representation is generated to meet Heights system requirements and UI interaction needs.

    5.X Adding a New Document Type to the Processing Pipeline

This section describes the end-to-end process, system touchpoints, effort estimation, and required resources for onboarding a new document type into the document processing pipeline.

5.X.1 Overview

Onboarding a new document type requires coordinated updates across multiple layers of the system, including:

Classification (document identification)
Fan-out routing (queue or SQL path)
Extraction (Document Intelligence / OCR / LLM)
Normalization (mapping to Heights schema)
Validation and persistence

The process ensures that the new document type is fully integrated into the pipeline and produces outputs consistent with existing system standards.

5.X.2 End-to-End Flow Impact

The following system components are impacted when adding a new document type:

Layer	Component	Change Required
Ingestion	API + Blob	No change
Classification	DI classifier	Add new label
Routing	Fan-out logic	Add routing rule
Extraction	Function App	Add new extractor
Normalization	Mapping layer	Add UI field mapping
Persistence	SQL schema	No structural change (schema-stable)
Aggregation	Fan-in	Automatically supported
5.X.3 Detailed Implementation Steps
Step 1: Classification Configuration

The document must first be identifiable by the classification module.

Activities
Add new label to Document Intelligence classifier
Upload sample documents (recommended: 10–20 samples minimum)
Train and publish classifier model
Update classification thresholds if required
Output
"doc_type": "NEW_DOC_TYPE"
📌 CCD Tip

Add a small table here:

Field	Value
Model Type	Custom Classifier
Training Samples	10–20
Confidence Threshold	e.g., 0.7
Step 2: Routing (Fan-Out Update)

Update the fan-out logic to route the new document type.

Decision Point
Condition	Routing Action
Extractable document	Send to Service Bus queue
Non-extractable	Store in SQL staging
Implementation
Add new doc_type condition
Map to:
existing queue OR
create new queue (if domain-specific)
Example
if doc_type == "NEW_DOC_TYPE":
    send_to_queue("new-doc-queue")
Output
Message emitted to queue OR
Record inserted into SQL
Step 3: Extraction Implementation

The extraction module is responsible for generating structured fields.

Options
Approach	When to Use
Prebuilt Model	Standard documents (ID, passport, etc.)
Custom DI Model	Semi-structured documents
OCR + LLM	Complex or highly variable documents
Activities
Configure Azure Document Intelligence model
Implement extractor function
Map raw output → extracted_fields
Output Format
"extracted_fields": {
  "FieldName": {
    "value": "...",
    "confidence": 90
  }
}
Step 4: Normalization (Heights Mapping)

Transform extracted data into Heights-compatible schema.

Transformation Types
Field mapping (rename fields)
Data formatting (dates, casing)
Structural changes (e.g., name splitting)
Null/default handling
Output
"ui_display_fields": {
  "FieldName": {
    "value": "...",
    "confidence": 90,
    "edited_sw": "N",
    "field_type": "STRING"
  }
}
📌 Add Table Here
Extracted Field	UI Field	Transformation
FullName	FirstName / LastName	Split
DOB	Date Of Birth	Format YYYY-MM-DD
Step 5: Validation Rules

Define validation logic for new document fields.

Examples
Mandatory fields
Confidence thresholds
Value format validation
Optional
Flag low-confidence fields for HITL review
Step 6: Testing & Validation
Testing Types
Type	Description
Unit Testing	Extractor + normalization
Integration Testing	Full pipeline
UAT	Business validation
Validation Criteria
Classification accuracy ≥ threshold
Field-level extraction accuracy
UI mapping correctness
No pipeline failures
Step 7: Deployment & Monitoring
Deployment
Deploy Function App changes
Update configuration (queues, env vars)
Monitoring
Track:
failure rates
confidence scores
processing latency
5.X.4 Estimated Effort
Task	Effort
Classification setup	0.5 – 1 day
Routing update	0.5 day
Extraction (prebuilt)	1 – 2 days
Extraction (custom model)	3 – 5 days
Normalization	1 – 2 days
Testing	1 – 2 days
Deployment	0.5 day
Total Effort
Scenario	Duration
Prebuilt model	3 – 5 days
Custom model	5 – 10 days
5.X.5 Required Resources
Technical Resources
Resource	Purpose
Azure Document Intelligence	Classification + extraction
Azure Service Bus	Queue routing
Azure Blob Storage	Document storage
Azure SQL	Intermediate storage
Azure Functions	Processing logic
Human Resources
Role	Responsibility
AI/ML Engineer	Model + extraction
Backend Engineer	Routing + normalization
QA Engineer	Testing
5.X.6 Dependencies & Risks
Availability of training data
Document variability (format/layout)
Model accuracy and confidence levels
Schema alignment with Heights
Queue capacity and scaling
5.X.7 Summary

Onboarding a new document type requires updates across classification, routing, extraction, and normalization layers. The system is designed to be modular, allowing new document types to be integrated with minimal impact on existing workflows while maintaining schema consistency and processing reliability.


dwqfdasSDASDFDAS------------------------------------------------------------------------------------------------------

Here are the two CCD-ready sections, aligned to your style and flow 👇

5.4.4.1 Prebuilt Processor (Passport, Driver License, SSN)

The Identity module leverages Azure Document Intelligence prebuilt models to process standard identity documents such as Passport, Driver License (DL), and Social Security Number (SSN) documents.

Processing Approach
Uses Azure Document Intelligence prebuilt-idDocument model
No custom training required
Designed for standardized, well-structured identity documents
Provides high-confidence extraction for common identity fields
Supported Document Types
Document Type	Model Used
Passport	Prebuilt ID Model
Driver License (DL)	Prebuilt ID Model
SSN	Prebuilt ID Model
Extraction Flow
Input document is passed to the prebuilt model
Model extracts key identity fields such as:
Name
Date of Birth
Document Number
Expiration Date
Issuing Authority
Output is returned in structured format and mapped to:
extracted_fields (raw model output)
ui_display_fields (normalized for Heights)
Output Characteristics
High confidence scores for standardized layouts
Minimal preprocessing required
Consistent field structure across documents
Direct compatibility with normalization layer
Advantages
Fast onboarding (no training required)
High accuracy for supported formats
Reduced development effort
Stable schema output
Limitations
Limited flexibility for non-standard layouts
May not capture custom or region-specific fields
Dependent on document quality and format consistency
Summary

Prebuilt processors are used for standard identity documents to enable rapid, reliable extraction with minimal configuration, ensuring consistent outputs aligned with the Heights schema.

5.4.4.2 Custom Processing for Birth Certificate

Birth Certificates are processed using a custom extraction approach, as they are highly variable in format and not fully supported by prebuilt models.

Processing Approach
Uses a combination of:
OCR (Document Intelligence / Read model)
LLM-based extraction (if required)
Custom parsing logic is implemented to extract required fields
Challenges with Birth Certificates
No standardized global format
Variability across:
countries
states
issuing authorities
Mixed layouts (tabular, paragraph, handwritten)
Extraction Flow
Document is processed using OCR to extract raw text
Extracted text is passed to:
rule-based parsing OR
LLM-based extraction
Key fields are identified and structured into:
extracted_fields
ui_display_fields
Typical Extracted Fields
Full Name
Date of Birth
Place of Birth
Gender
Parent Names (if available)
Registration Number (if present)
Normalization

Due to variability, normalization includes:

Standardizing date formats
Mapping fields to Heights schema
Handling missing or optional fields
Assigning confidence scores (derived if needed)
Output Characteristics
Flexible extraction logic
Handles diverse layouts
May include lower confidence for ambiguous fields
Supports fallback and reprocessing
Effort Consideration
Component	Effort
OCR setup	Low
Extraction logic (rules/LLM)	Medium
Normalization	Medium
Testing (multiple formats)	High
Advantages
Supports non-standard documents
Flexible and extensible
Can adapt to new formats without retraining DI models
Limitations
Higher implementation effort
Requires extensive testing
May have variable confidence levels
Summary

Birth Certificates are processed using a custom extraction pipeline to handle variability in format and structure, ensuring that required identity fields are captured and normalized for downstream processing.

If you want, I can next:

connect this with fan-out + identity queue routing (very important linkage)
or refine confidence + HITL logic specifically for identity docs 👍


[-------------------------------------------------------------------------
                                                                                                                                                                                     
7. Deployment Strategy
7.1 Overview

The Document Processing System is deployed using Azure Resource Manager (ARM) templates to ensure consistent, repeatable, and environment-independent infrastructure provisioning.

The ARM template recreates the complete infrastructure currently hosted in the Deloitte Azure sandbox within the target DHHS Azure environment.

Post-deployment steps include application configuration, database setup, and AI model training.

7.2 Infrastructure Deployment using ARM Template

The ARM template provisions the following Azure resources:

7.2.1 Compute & Application Layer
Azure Function App (nh-docproc-functionapp-premium)
App Service Plan (ASP-NHDocumentAIPOCworkRG)

These components host the backend processing logic, including:

Document ingestion APIs
Classification orchestration
Extraction pipelines
7.2.2 Storage Layer
Azure Storage Account (nhinitialpdfstorage) – stores raw uploaded documents
Azure Storage Account (nhclassificationdata) – stores classification-related artifacts
Additional storage (nhdocumentaipocworkb153) – used for intermediate processing / staging
7.2.3 Messaging Layer
Azure Service Bus Namespace (nh-sb-blob-to-classification)

Used for:

Asynchronous document processing
Queue-based decoupling between ingestion, classification, and extraction modules
7.2.4 Database Layer
Azure SQL Server (sqlinitialpdfmeta)
Azure SQL Database (T001_INITIAL_PDF_METADATA)

Used for:

Storing document metadata
Tracking processing status
Supporting UI retrieval and prefill workflows
7.2.5 Monitoring & Observability
Application Insights (nh-docproc-ai-insights)
Log Analytics Workspace

Used for:

Logging
Performance monitoring
Failure tracking
Debugging production issues
7.2.6 AI / Document Processing Layer
Azure AI Document Intelligence Resource (NH-1744-Doc-AI-Extractor)

Used for:

OCR (Read API)
Prebuilt models (Pay Stub, ID, etc.)
Custom classification and extraction models
7.3 Deployment Strategy
Primary Approach: ARM Template Deployment
Infrastructure is deployed using a parameterized ARM template
Supports multiple environments (Dev, UAT, Prod)
Ensures consistency across deployments
Reduces manual setup errors
Secondary Approach: Manual Deployment (Fallback)

Used when:

Minor resource changes are required
Debugging environment-specific issues
Testing incremental updates
7.4 Post-Deployment Configuration

After ARM deployment, the system is not fully functional until the following configurations are completed:

7.4.1 Application Configuration
Configure Function App settings:
Storage connection strings
Service Bus connection strings
SQL connection string
Document Intelligence endpoint and key
OpenAI / LLM configurations (if used)
Set environment variables securely (preferably via Key Vault)
7.4.2 Function App Code Deployment
Deploy Python backend code to Function App
Ensure:
Dependencies are installed
Runtime version matches development environment
API endpoints are accessible
7.4.3 Database Setup
Create required tables and schema in SQL Database
Apply:
Metadata tables
Status tracking tables
Any indexing required for performance
7.4.4 Messaging Configuration
Create Service Bus queues/topics:
Ingestion queue
Classification queue
Extraction queue
Configure:
Retry policies
Dead-letter queues (DLQ)
7.4.5 Storage Configuration
Create required containers:
Raw documents container
Processed documents container
Temporary staging container
Set access policies and lifecycle rules if required
7.5 AI Model Setup and Training

The ARM template creates the Document Intelligence resource, but does not include trained models.

The following steps must be performed:

7.5.1 Custom Classification Model
Upload labeled training dataset
Train classifier to identify document types
Validate model accuracy
Publish model and capture model ID
7.5.2 Custom Extraction Models

For each document type:

Upload labeled datasets
Train extraction models
Validate extracted fields
Publish models and store model IDs
7.5.3 Model Configuration
Update application configuration with:
Classifier model ID
Extractor model IDs
Map models to document types in processing logic
7.6 Validation and Testing

After deployment and configuration:

Upload test documents
Validate:
Classification accuracy
Extraction accuracy
End-to-end workflow
Verify:
Data stored in SQL
UI prefill functionality
Error handling and retries
7.7 Security and Access Configuration
Configure:
Role-based access control (RBAC)
Managed identities (if used)
Key Vault integration for secrets
Ensure:
No hardcoded credentials
Proper access isolation per environment
7.8 Summary

The ARM template provisions the full infrastructure stack required for the Document Processing System. However, additional steps such as application deployment, database setup, and AI model training are required to make the system fully operational.

This approach ensures:

Scalability
Repeatability
Environment consistency
Production readiness
                                                                                                                                                                                     
                                                                                                                                                                                     

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

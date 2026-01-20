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


quad = _find_largest_rectangle(img)


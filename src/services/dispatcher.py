from services.classification.google_docai import classify_document
from services.processors.bank_statement import process_bank_statement
from services.processors.pay_stub import process_pay_stub
from src.services.mapping.bank_statement_mapper import map_azure_to_custom_transactions

def route_document(file_bytes: bytes, filename: str) -> dict:
    doc_type = classify_document(file_bytes, filename)

    if doc_type == "bank_statement":
        raw_output = process_bank_statement(file_bytes, filename)
        transactions = map_azure_to_custom_transactions(raw_output)
        return {"transactions": transactions}

    elif doc_type == "pay_stub":
        raw_output = process_pay_stub(file_bytes, filename)
        # TODO: Add mapping logic when pay_stub_mapper.py is ready
        return {"transactions": raw_output.get("transactions", [])}

    else:
        return {
            "status": "unsupported_document",
            "doc_type": doc_type,
            "transactions": []
        }

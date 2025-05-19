def classify_document(file_bytes: bytes, filename: str) -> str:
    # TODO: Replace this mock logic with real Google Doc AI classification
    if "bank" in filename.lower():
        return "bank_statement"
    elif "pay" in filename.lower():
        return "pay_stub"
    return "unknown"

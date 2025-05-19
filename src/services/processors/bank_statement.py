from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from io import BytesIO
import json

# NOTE: Keep these in environment variables in production
AZURE_ENDPOINT = "https://venka.cognitiveservices.azure.com/"
AZURE_KEY = "FmA8C0l5UHynz2tNCdA2gMXmLJAOsCpjp2eWncyiPvHzHgAPe3QbJQQJ99BEACYeBjFXJ3w3AAALACOGYitX"

client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

def process_bank_statement(file_bytes: bytes, filename: str) -> dict:
    try:
        poller = client.begin_analyze_document("prebuilt-bankStatement.us", body=BytesIO(file_bytes))
        result = poller.result()

        # Extract structured fields
        fields = {}

        for idx, statement in enumerate(result.documents):
            def safe_extract(field, field_type="string"):
                if not field:
                    return None
                return getattr(field, f"value_{field_type}")

            fields["AccountHolderName"] = safe_extract(statement.fields.get("AccountHolderName"))
            fields["BankName"] = safe_extract(statement.fields.get("BankName"))
            fields["AccountHolderAddress"] = safe_extract(statement.fields.get("AccountHolderAddress"), "address")
            fields["BankAddress"] = safe_extract(statement.fields.get("BankAddress"), "address")
            fields["StatementStartDate"] = safe_extract(statement.fields.get("StatementStartDate"), "date")
            fields["StatementEndDate"] = safe_extract(statement.fields.get("StatementEndDate"), "date")

            accounts_data = []
            accounts = statement.fields.get("Accounts")
            if accounts:
                for account in accounts.value_array:
                    acc_obj = account.value_object
                    account_info = {
                        "AccountNumber": safe_extract(acc_obj.get("AccountNumber")),
                        "AccountType": safe_extract(acc_obj.get("AccountType")),
                        "BeginningBalance": safe_extract(acc_obj.get("BeginningBalance"), "number"),
                        "EndingBalance": safe_extract(acc_obj.get("EndingBalance"), "number"),
                        "TotalServiceFees": safe_extract(acc_obj.get("TotalServiceFees"), "number"),
                        "Transactions": [],
                    }

                    transactions = acc_obj.get("Transactions")
                    if transactions:
                        for txn in transactions.value_array:
                            txn_obj = txn.value_object
                            account_info["Transactions"].append({
                                "Date": safe_extract(txn_obj.get("Date"), "date"),
                                "Description": safe_extract(txn_obj.get("Description")),
                                "DepositAmount": safe_extract(txn_obj.get("DepositAmount"), "number"),
                                "WithdrawalAmount": safe_extract(txn_obj.get("WithdrawalAmount"), "number"),
                            })
                    accounts_data.append(account_info)

            fields["Accounts"] = accounts_data

        return {
            "status": "Processed by Azure",
            "filename": filename,
            "extracted_fields": fields
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}



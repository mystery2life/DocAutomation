import re

def map_azure_to_custom_transactions(raw_data: dict) -> list:
    fields = raw_data.get("extracted_fields", {})
    name = fields.get("AccountHolderName")
    institution = fields.get("BankName")
    transactions = []

    for acc in fields.get("Accounts", []):
        account_number = acc.get("AccountNumber")
        account_type = acc.get("AccountType", None)

        for txn in acc.get("Transactions", []):
            desc = txn.get("Description", "")
            date = txn.get("Date")
            deposit = txn.get("DepositAmount")
            withdrawal = txn.get("WithdrawalAmount")

            # Transaction value = deposit or withdrawal (positive)
            value = deposit if deposit is not None else withdrawal

            # Simple rule to extract check number from description
            check_match = re.search(r'check\s*#?\s*(\d{3,})', desc, re.IGNORECASE)
            check_number = check_match.group(1) if check_match else None

            transactions.append({
                "name": name,
                "institution_name": institution,
                "transaction_type": desc,
                "institution_number": account_number,
                "account_type": account_type,
                "check_number": check_number,
                "transaction_date": date,
                "transaction_value": value,
            })

    return transactions

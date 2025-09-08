import streamlit as st
import requests
import base64
import re
from dateutil import parser

# --- Streamlit config ---
st.set_page_config(page_title="Pay Stub Processor", layout="wide")
st.title("Pay Stub Document Processing")

# --- Session Initialization ---
for key in ["extracted_data", "edited_fields"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key == "edited_fields" else None

# --- Document Metadata Input ---
st.subheader("Document Metadata")
rid = st.text_input("RID Number")
doc_id = st.text_input("Document ID")
full_name = st.text_input("Full Name")
doc_type = st.selectbox("Document Type", ["pay_stub", "bank_statement"])
model_type = st.radio("Model Option", ["azure_query", "azure_llm", "google_docai"])

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload Pay Stub or Bank Statement (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    # --- Document Preview ---
    st.subheader("Document Preview")
    if uploaded_file.type == "application/pdf":
        b64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800px"></iframe>', unsafe_allow_html=True)
    else:
        st.image(file_bytes, caption=uploaded_file.name)

    # --- Process Document Button ---
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            res = requests.post(
                "http://localhost:8000/api/v1/process-document",
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                data={"filename": uploaded_file.name, "model_type": model_type}
            )
            if res.status_code == 200:
                response_json = res.json()

                # Inject metadata
                extracted_fields = response_json.get("extracted_fields", {})
                extracted_fields["RID"] = {"value": rid, "confidence": 100}
                extracted_fields["DocumentID"] = {"value": doc_id, "confidence": 100}
                extracted_fields["FullName"] = {"value": full_name, "confidence": 100}

                response_json["extracted_fields"] = extracted_fields
                st.session_state.extracted_data = response_json
                st.session_state.edited_fields = {}  # reset

                st.success("Document processed successfully.")
            else:
                st.error("Document processing failed.")

# --- Field Normalization ---
def normalize_field(key, val):
    value = val.get("value", "")
    if value is None:
        return ""
    try:
        if "Pay" in key and isinstance(value, str) and "$" in value:
            cleaned = re.sub(r"[^\d.]", "", value)
            return f"{float(cleaned):.2f}"
        elif "Date" in key:
            return parser.parse(str(value)).strftime("%Y-%m-%d")
        elif "Hours" in key:
            cleaned = re.sub(r"[^\d.]", "", str(value))
            return f"{float(cleaned):.2f}"
        elif "SSN" in key and isinstance(value, str) and "-" in value:
            return "***-**-" + value[-4:]
        else:
            return str(value).strip()
    except Exception:
        return str(value).strip() if value else ""

# --- Editable UI Output ---
if st.session_state.extracted_data:
    st.subheader("✏️ Editable Fields")
    extracted = st.session_state.extracted_data.get("extracted_fields", {})

    # ---- Add custom fields here ----
    extracted["PaymentType"] = {"value": "regular", "confidence": 100}
    extracted["GrossAmount"] = extracted.get("CurrentPeriodGrossPay", {"value": "", "confidence": None})

    # PayFrequency logic
    try:
        start_date = parser.parse(extracted.get("PayPeriodStartDate", {}).get("value", ""))
        end_date = parser.parse(extracted.get("PayPeriodEndDate", {}).get("value", ""))
        days_diff = (end_date - start_date).days

        if days_diff <= 8:
            frequency = "Weekly"
        elif days_diff <= 15:
            frequency = "Bi-Weekly"
        elif days_diff <= 20:
            frequency = "Semi-Monthly"
        else:
            frequency = "Monthly"

        extracted["PayFrequency"] = {"value": frequency, "confidence": 100}
    except Exception:
        extracted["PayFrequency"] = {"value": "", "confidence": None}

    required_keys = [
        "FullName", "RID", "DocumentID", "PayRate", "HoursWorked", "JobTitle",
        "EmployeeName", "EmployerName", "EmployerAddress", "PayDate",
        "PayPeriodStartDate", "PayPeriodEndDate",
        "PaymentType", "GrossAmount", "PayFrequency"
    ]

    # Show fields in columns of 3
    cols = st.columns(3)
    for idx, key in enumerate(required_keys):
        field_data = extracted.get(key, {"value": "", "confidence": None})
        value = normalize_field(key, field_data)
        conf = field_data.get("confidence", 0)

        with cols[idx % 3]:
            st.markdown(f"<label style='font-weight:bold;'>{key}</label>", unsafe_allow_html=True)
            st.session_state.edited_fields[key] = st.text_input(
                label=f"{key}",
                value=value,
                key=f"{key}_input",
                label_visibility="collapsed",
            )

            if not value:
                st.markdown("<span style='color:red; font-size:13px;'>❌ Missing field</span>", unsafe_allow_html=True)
            elif conf is not None and conf < 75:
                st.markdown(f"<span style='color:red; font-size:13px;'>⚠️ Low Confidence: {conf:.2f}%</span>", unsafe_allow_html=True)
            else:
                st.caption(f"Confidence: {conf:.2f}%")

    if st.button("💾 Save"):
        st.success("📝 Changes saved (in session).")
        st.json(st.session_state.edited_fields)


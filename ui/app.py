import streamlit as st
import requests
import base64
import pandas as pd

st.set_page_config(page_title="Doc Classifier", layout="centered")
st.title("📄 Document Classifier")

# Store session state for classification
if "classification" not in st.session_state:
    st.session_state.classification = None
if "transactions" not in st.session_state:
    st.session_state.transactions = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.subheader("Preview")
    file_bytes = uploaded_file.read()

    if uploaded_file.type == "application/pdf":
        b64_pdf = base64.b64encode(file_bytes).decode()
        st.markdown(f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500"></iframe>', unsafe_allow_html=True)
    else:
        st.image(file_bytes, caption=uploaded_file.name)

    # Step 1: Classify
    if st.button("🔍 Classify Document"):
        res = requests.post(
            "http://localhost:8000/api/v1/classify-document",
            files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
            data={"filename": uploaded_file.name}
        )
        if res.ok:
            data = res.json()
            st.session_state.classification = data["doc_type"]
            st.success(f"Type: {data['doc_type']} (Confidence: {data['confidence']:.2f})")
        else:
            st.error("Classification failed.")
            st.session_state.classification = None

    # Step 2: Extract (only show if classified as bank_statement or pay_stub)
    if st.session_state.classification in ["bank_statement", "pay_stub"]:
        if st.button("📤 Extract Transactions"):
            res = requests.post(
                "http://localhost:8000/api/v1/process-document",
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
            )
            if res.ok:
                response_json = res.json()
                st.write("🔎 Raw Response:", response_json)  # Add this line
                st.session_state.transactions = response_json.get("transactions", [])[:10]
                st.success("Extraction successful! Showing top 10 transactions.")
                st.dataframe(pd.DataFrame(st.session_state.transactions))
            else:
                st.error("Extraction failed.")


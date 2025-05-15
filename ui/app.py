import streamlit as st
import requests
import base64

st.set_page_config(page_title="Doc Classifier", layout="centered")

st.title("📄 Document Classifier")

uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.subheader("Preview")
    file_bytes = uploaded_file.read()

    if uploaded_file.type == "application/pdf":
        b64_pdf = base64.b64encode(file_bytes).decode()
        st.markdown(f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500"></iframe>', unsafe_allow_html=True)
    else:
        st.image(file_bytes, caption=uploaded_file.name)

    if st.button("🔍 Classify Document"):
        res = requests.post(
            "http://localhost:8000/api/v1/classify-document",
            files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
        )
        if res.ok:
            data = res.json()
            st.success(f"Type: {data['doc_type']} (Confidence: {data['confidence']:.2f})")
        else:
            st.error("Classification failed.")

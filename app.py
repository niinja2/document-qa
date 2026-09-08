import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("Document QA")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
)

if uploaded_file is not None:
    if st.button("Upload"):
        session_id = str(uuid.uuid4())
        with st.spinner("Uploading and processing document..."):
            response = requests.post(
                f"{API_URL}/upload",
                data={"session_id": session_id},
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
            )
        if response.status_code == 200:
            st.session_state.session_id = session_id
            st.success("Document uploaded successfully.")
        else:
            try:
                detail = response.json().get("detail", "Unknown error")
            except Exception:
                detail = response.text or "Unknown error"
            st.error(f"Upload failed: {detail}")

if "session_id" in st.session_state:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask",
                data={
                    "session_id": st.session_state.session_id,
                    "question": question,
                },
            )
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.markdown(f"**Answer:** {answer}")
        else:
            try:
                detail = response.json().get("detail", "Unknown error")
            except Exception:
                detail = response.text or "Unknown error"
            st.error(f"Error: {detail}")

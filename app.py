import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("Document QA")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Upload"):
        session_id = str(uuid.uuid4())
        files = [
            ("files", (f.name, f, f.type))
            for f in uploaded_files
        ]
        with st.spinner("Uploading and processing documents..."):
            response = requests.post(
                f"{API_URL}/upload",
                data={"session_id": session_id},
                files=files,
            )
        if response.status_code == 200:
            st.session_state.session_id = session_id
            st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")
        else:
            try:
                detail = response.json().get("detail", "Unknown error")
            except Exception:
                detail = response.text or "Unknown error"
            st.error(f"Upload failed: {detail}")

if "session_id" in st.session_state:
    question = st.text_input("Ask a question about the uploaded documents")
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

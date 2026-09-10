import uuid
import time
import requests
import streamlit as st
from config import API_URL


@st.cache_resource
def wait_for_api():
    for _ in range(30):
        try:
            requests.get(f"{API_URL}/openapi.json", timeout=1)
            return True
        except requests.RequestException:
            time.sleep(2)
    return False


st.title("Document QA")

with st.spinner("API loading..."):
    if not wait_for_api():
        st.error("API not reachable. Please refresh.")
        st.stop()

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
    with st.form("ask_form"):
        question = st.text_input("Ask a question about the document")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
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

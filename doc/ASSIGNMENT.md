# AI Backend Developer — Assignment 1: AI-driven Document Insight Service

## Description

Build a Python-based REST API (FastAPI or Flask) that:
- Ingests PDF or image documents (e.g., scanned contracts, invoices) and extracts text (via EasyOCR, PyMuPDF, or both, or other)
- Answers user questions about the uploaded documents using a QA model (e.g., DistilBERT QA or any other transformer, local or any LLM via API)
- Find or create random/dummy test docs (store in Github repo)

---

## Core Requirements

- `POST /upload` — accept one or more documents, store them for session-based or user-based retrieval
- `POST /ask` — take a question, run a QA pipeline over the stored documents, and return the answer
- Dockerize the entire application

---

## Optional Enhancements (pick at least one)

- Integrate Named Entity Recognition (choose model) to highlight entities in responses
- Implement Retrieval-Augmented Generation (RAG) with embeddings (FAISS or similar) for improved context retrieval
- Add a caching layer (Redis or other) to store and quickly fetch document embeddings

## General Optional Enhancements (pick any)

- **UI**: Prototype with Gradio or Streamlit to demonstrate the workflow
- **MLOps**: Modular packaging, CI/CD integration, structured logging, monitoring
- **Performance**: Profile model inference latency or API throughput and optimize
- **Security**: JWT authentication, input sanitization, rate limiting

---

## LLM Usage

You may use ChatGPT or any LLM to accelerate development. Be ready to explain how you validated AI-generated code, and why you chose the final approach.

---

## Delivery

Host your solution in a GitHub repository. `README.md` must include:
- Setup instructions
- Manual installation and Docker option instructions
- Example API requests and responses
- Brief description of your approach (tools/models chosen and why)
- Optional: screenshots, demo video, or other creative additions

Any extra creative feature beyond the above is a big plus.

---

## Evaluation Criteria

| Criterion | What they look at |
|---|---|
| **Functionality** | Does the solution do what was asked? Does it cover edge cases? |
| **Code quality** | Readability, structure, naming, organisation, maintainability |
| **Architecture & design** | Soundness of chosen approaches, separation of concerns, scalability, reasoning behind decisions |
| **AI/LLM work** | Prompt design, output handling, error handling, model limitations |
| **Decisions & trade-offs** | What was consciously decided, what was left out and why (time is limited) |
| **Documentation** | Short README, setup instructions, explanation of approach |

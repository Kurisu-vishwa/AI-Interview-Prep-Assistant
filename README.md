# AI-Interview-Prep-Assistant
A Retrieval-Augmented Generation (RAG) based AI assistant for Machine Learning interview preparation.

This project enables users to ask ML interview questions conversationally, retrieve grounded answers from a custom knowledge base, upload new documents dynamically, and maintain short-term conversational memory for follow-up questions.

---

## Features

- Semantic retrieval using ChromaDB  
- Context-grounded answers using Groq LLM  
- Conversational memory using Buffer Window Memory  
- Dynamic document upload and incremental ingestion  
- Persistent vector database  
- Streamlit chat interface  
- Strict `INSUFFICIENT_CONTEXT` fallback when answer is not found in retrieved documents  

---

## Tech Stack

- Python  
- LangChain  
- ChromaDB  
- HuggingFace Embeddings (`BAAI/bge-base-en-v1.5`)  
- Groq LLM (`llama-3.1-8b-instant`)  
- Streamlit  

---
## How It Works

1. Documents are loaded and split into chunks  
2. Chunks are embedded using HuggingFace embeddings  
3. Embeddings are stored in ChromaDB  
4. User queries retrieve relevant chunks using MMR retrieval  
5. Groq LLM answers using retrieved context only  
6. New documents can be uploaded dynamically through the UI and added to the vector store without rebuilding the entire database  

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/rag-ai-interview-prep-assistant.git
cd rag-ai-interview-prep-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `keys.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Queries

Base knowledge:

- What is bagging?  
- Difference between bagging and boosting?  
- What is overfitting?  

Memory follow-up:

- Which one reduces variance?  

Dynamic upload demo:

Upload a document containing a new concept:

- What is Label Smoothing?  

---

## Example Workflow

Question not in knowledge base:

```text
User: What is Label Smoothing?
Assistant: INSUFFICIENT_CONTEXT
```

Upload new document containing Label Smoothing.

Ask again:

```text
User: What is Label Smoothing?
Assistant: Uses softer targets like 0.9 and 0.1 to improve generalization.
```

---
# Live demo
Demo link: https://huggingface.co/spaces/Akira10001/AI-Interview-Prep-Assistant

# Self-RAG

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain and LangGraph, implemented as a series of Jupyter notebooks.

## What is Self-RAG?

Self-RAG (Self-Reflective Retrieval-Augmented Generation) is an advanced RAG architecture that improves response quality through **self-reflection** and **iterative refinement**. Unlike traditional RAG systems that retrieve once and generate, Self-RAG introduces reflection tokens that allow the model to:

### Core Concepts

1. **Retrieve**: Query the vector store to find relevant document chunks
2. **Generate**: Produce an initial answer using retrieved context
3. **Reflect**: Critically evaluate the generated response against criteria:
   - **Relevance**: Does the retrieved context actually support the answer?
   - **Accuracy**: Is the answer factually correct based on the source?
   - **Completeness**: Does the answer fully address the question?
4. **Refine**: If reflection identifies issues, re-retrieve or regenerate
5. **Output**: Final verified response

### How This Implementation Works

```
Question → Retrieve → Generate → Reflect → [If needed: Re-retrieve/Regenerate] → Final Answer
```

The LangGraph state machine orchestrates this flow:
- **State management**: Tracks question, retrieved documents, generated response, and reflection scores
- **Conditional edges**: Branches based on reflection outcomes (pass/fail thresholds)
- **Iterative refinement**: Can loop back for additional retrieval or regeneration cycles

### Benefits Over Standard RAG

| Standard RAG | Self-RAG |
|--------------|----------|
| Single retrieval | Dynamic re-retrieval |
| One-shot generation | Iterative refinement |
| No quality check | Self-reflection tokens |
| May hallucinate | Grounding verification |

This implementation uses OpenAI's GPT-4o-mini for both generation and reflection, with FAISS for efficient similarity search over chunked PDF documents.

## Project Structure

```
Self RAG/
├── step1.ipynb        # Initial setup and imports
├── step2.ipynb        # Document loading
├── step3.ipynb        # Text splitting
├── step4.ipynb        # Vector store creation
├── step5.ipynb        # Retriever setup
├── step6.ipynb        # Complete RAG pipeline with LangGraph
├── documents/         # PDF source documents
├── .env               # API keys and configuration 
```

## Documents

The RAG system indexes three PDF documents:
- `documents/Company_Policies.pdf`
- `documents/Company_Profile.pdf`
- `documents/Product_and_Pricing.pdf`

## Setup

### 1. Clone and Install

```bash
# Ensure Python environment with required packages
pip install langchain langchain-community langchain-openai langchain-core langgraph faiss-cpu pypdf2 python-dotenv
```

### 2. Configure Environment

Create a `.env` file with the following API keys:

```env
OPENAI_API_KEY="your-openai-key"
GOOGLE_API_KEY="your-google-key"
HUGGINGFACEHUB_API_TOKEN="your-hf-token"
MISTRAL_API_KEY="your-mistral-key"
LANGSMITH_TRACING="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langsmith-key"
LANGCHAIN_PROJECT="RAG_PIPELINE"
```

### 3. Add Documents

Place your PDF documents in the `documents/` directory.

## Usage

Run the notebooks sequentially or execute `step6.ipynb` for the complete pipeline:

```python
# In Jupyter
%run step6.ipynb
```

## Configuration

Default parameters:
- **Chunk size**: 600 characters
- **Chunk overlap**: 150 characters
- **Retriever k**: 4 (top results)
- **Embedding model**: `mistral-embed`
- **LLM model**: `mistral-small-latest`
- **Temperature**: 0

## Features

- Modular notebook progression for learning and debugging
- LangGraph state machine for RAG workflow
- FAISS vector store for efficient similarity search
- LangSmith tracing for observability
- Multi-document ingestion

## Notes

- `.env` is gitignored but required for execution
- Jupyter checkpoint files are auto-ignored
- All notebooks should run in sequence for full pipeline understanding
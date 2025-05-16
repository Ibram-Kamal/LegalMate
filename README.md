# Legal Mate (Streamlit RAG)

**Legal Mate** is a Streamlit-based Retrieval-Augmented Generation (RAG) application designed to analyze Arabic legal contracts. It accepts PDF/TXT uploads, splits the document into chunks, embeds them via **SentenceTransformer**, indexes with **FAISS**, and leverages Google’s **Gemini** (via the GenAI API) for contextual legal analysis in Arabic.

---

## Table of Contents

1. [Features](#features)
2. [Demo](#demo)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Architecture Overview](#architecture-overview)
8. [Code Structure](#code-structure)
9. [Customization & Extensions](#customization--extensions)
10. [Troubleshooting](#troubleshooting)
11. [Third-Party Services](#third-party-services)
12. [License](#license)

---

## Features

* **PDF & TXT Upload**: Accepts Arabic contracts as PDF or plain-text files.
* **Text Extraction**: Uses `pdfplumber` for accurate PDF text extraction.
* **Chunking**: Splits long contracts into overlapping chunks via `langchain.text_splitter.RecursiveCharacterTextSplitter`.
* **Embeddings & Indexing**: Creates multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and indexes with FAISS for fast similarity search.
* **Gemini Integration**: Streams responses from Google’s Gemini model (`gemini-2.5-flash-preview-04-17`) for legal analysis in Arabic.
* **Preset Actions**: Buttons for simplified summary, parties extraction, contract type, and detailed analysis.
* **Custom Queries**: Chat input for any user-specific legal question.
* **Session History**: Maintains chat history within the session for continuity.

---

## Demo

![image](https://github.com/user-attachments/assets/fa4a0f2c-f1c7-4afe-bbc3-f4aceb845cd4)


---

## Requirements

This project relies on the following Python packages:

```text
streamlit>=1.18.1
pdfplumber>=0.9.0
langchain>=0.0.300
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
numpy>=1.23.0
google-genai>=0.2.0
```

You can install them via:

```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ibram-kamal/legal-mate.git
   cd legal-mate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   streamlit run LegalMate_LV.py
   ```

---

## Configuration

1. **Set your API key**

   ```bash
   export GENAI_API_KEY="YOUR_GOOGLE_GENAI_KEY"
   ```

2. *(Optional)* Override the default Gemini model via environment variable:

   ```bash
   export GENAI_MODEL_ID="gemini-2.5-flash-preview-04-17"
   ```

3. **Verify** that your `GENAI_API_KEY` is valid and has access to the GenAI API.

---

## Usage

1. Upload an **Arabic** contract file (PDF or TXT).
2. Wait for chunking, embedding, and FAISS index building.
3. Choose one of the preset actions:

   * 🔍 تحليل مبسط للعقد
   * 👥 أطراف العقد
   * 📄 نوع العقد
   * 😨 تحليل تفصيلي للعقد
4. Or type a **custom question** in the chat input.
5. View streamed responses and session history on the right.

---

## Architecture Overview

```text
User → Streamlit UI
      ├─ Text Extraction (pdfplumber / txt)
      ├─ Chunking (LangChain)
      ├─ Embedding (SentenceTransformer)
      ├─ FAISS Index
User Query → Embedding → FAISS Retrieval → Context Assembly
      └─ Google Gemini (GenAI API) → Streamed Response → UI
```

---

## Code Structure

```
legal-mate/
├── README.md
├── requirements.txt
├── LegalMate_LV.py
└── utils/
    └── extract_text.py    # (if refactored)
```

* **LegalMate\_LV.py**: Main Streamlit application
* **requirements.txt**: Python dependencies
* **utils/extract\_text.py**: Optional helper for text extraction

---

## Customization & Extensions

* **Index Tuning**: Swap `IndexFlatL2` for `IndexIVFFlat` or HNSW for large documents.
* **Embedding Model**: Use Arabic-specialized models (e.g., AraBERT) for better legal nuance.
* **Prompt Engineering**: Adjust system prompts for different legal tasks or languages.
* **UI Enhancements**: Add “Clear History” button, progress bars, or PDF previews.

---

## Troubleshooting

* **“No API Key” Error**: Ensure `GENAI_API_KEY` is set and non-empty.
* **PDF Extraction Errors**: Confirm the PDF is text-based (not scanned); consider OCR.
* **Slow Performance**: Limit retrieval to top-k chunks:

  ```python
  distances, indices = index.search(q_vec, k=10)
  ```

---

## Third-Party Services

* **Google Gemini API**
  This application uses Google’s Gemini model via the GenAI API. Usage is subject to Google Cloud’s Terms of Service, quotas, and pricing. Manage your API key, usage limits, and associated costs.

---



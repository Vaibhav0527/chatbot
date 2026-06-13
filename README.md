# Assignment: Build a RAG Chatbot with LangChain

This project implements a local PDF-based RAG chatbot using:
- LangChain
- FAISS (local vector store)
- HuggingFace embeddings
- Google Gemini as the LLM
- Optional Streamlit UI

## 1) Setup

Create and activate a virtual environment:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your environment file:

```powershell
copy .env.example .env
```

In `.env`, set one of:
- `GOOGLE_API_KEY`

If you need to override the default model, use `GOOGLE_MODEL=gemini-2.5-flash`.

## 2) Load + Split + Embed + Store + Retrieve + Generate

Run terminal chatbot:

```powershell
python -m rag_chatbot.cli --pdf "path\to\your_notes.pdf"
```

This command will:
1. Load your PDF via `PyPDFLoader`
2. Split text with `RecursiveCharacterTextSplitter`
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`
4. Build a local FAISS index
5. Run retrieval + generation chain for QA

If answer is missing in context, bot is instructed to return exactly:

`Not found in document.`

## 3) Bonus: Streamlit UI

```powershell
streamlit run streamlit_app.py
```

Upload a PDF, then ask questions.

## 4) Suggested 5 Test Questions

Use questions that include:
1. A direct fact from the PDF
2. A definition from the PDF
3. A multi-sentence explanation in the PDF
4. A question not covered in the PDF (should return `Not found in document.`)
5. A page-specific detail from the PDF

## 5) Submission Checklist

- Source code in GitHub repo
- README with your pipeline explanation
- Screenshots or short demo video
- Include at least 5 tested questions and outputs

## Project Structure

```text
.
├── rag_chatbot/
│   ├── __init__.py
│   ├── cli.py
│   ├── pipeline.py
│   └── providers.py
├── .env.example
├── .gitignore
├── requirements.txt
├── streamlit_app.py
└── README.md
```

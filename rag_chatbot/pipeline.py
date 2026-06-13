from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chatbot.providers import build_chat_model

SYSTEM_PROMPT = """You are a strict QA assistant.
Answer only from the provided context.
If the answer is missing, unclear, or cannot be supported by the context, reply exactly:
Not found in document.
Do not use outside knowledge.
Keep the answer concise.
"""


@dataclass
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_k: int = 4
    score_threshold: float = 0.35


class PDFRAG:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._vectorstore: FAISS | None = None
        self._chain: Any | None = None

    def build_from_pdf(self, pdf_path: str | Path) -> None:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self._vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.config.similarity_k,
                "score_threshold": self.config.score_threshold,
            },
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{input}",
                ),
            ]
        )

        qa_chain = create_stuff_documents_chain(build_chat_model(), prompt)
        self._chain = create_retrieval_chain(retriever, qa_chain)

    def answer(self, question: str) -> str:
        if self._chain is None:
            raise RuntimeError("Index not built. Call build_from_pdf() first.")

        response = self._chain.invoke({"input": question})
        return response.get("answer", "Not found in document.")

    def save_index(self, output_dir: str | Path) -> None:
        if self._vectorstore is None:
            raise RuntimeError("Vector store not available. Build index first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(output_dir))

    def load_index(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self._vectorstore = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

        retriever = self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.config.similarity_k,
                "score_threshold": self.config.score_threshold,
            },
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{input}",
                ),
            ]
        )

        qa_chain = create_stuff_documents_chain(build_chat_model(), prompt)
        self._chain = create_retrieval_chain(retriever, qa_chain)

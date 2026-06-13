from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_chatbot.pipeline import PDFRAG

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="P", layout="wide")
st.title("PDF RAG Chatbot")
st.caption("Ask questions from your uploaded PDF only.")

if "rag" not in st.session_state:
    st.session_state.rag = None

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded is not None:
    temp_path = Path("uploaded_document.pdf")
    temp_path.write_bytes(uploaded.read())

    with st.spinner("Building index from PDF..."):
        rag = PDFRAG()
        rag.build_from_pdf(temp_path)
        st.session_state.rag = rag

    st.success("Index built. You can ask questions now.")

question = st.text_input("Your question")
if st.button("Ask") and question:
    if st.session_state.rag is None:
        st.warning("Upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.answer(question)
        st.markdown("### Answer")
        st.write(answer)

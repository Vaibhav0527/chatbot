import argparse

from dotenv import load_dotenv

from rag_chatbot.pipeline import PDFRAG


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with your PDF using RAG.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument(
        "--save-index",
        default=".faiss_index",
        help="Directory to save FAISS index after building",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    rag = PDFRAG()
    rag.build_from_pdf(args.pdf)
    rag.save_index(args.save_index)

    print("RAG chatbot is ready. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = rag.answer(question)
        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()

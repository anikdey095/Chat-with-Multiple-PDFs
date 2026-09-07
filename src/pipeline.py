"""
Full RAG Pipeline — সব module কে একসাথে জোড়া দেয়।
"""
from loader import load_document
from chunker import chunk_text
from embedder import get_collection, index_chunks, search
from generator import generate_answer


def build_index(file_path: str, source_name: str = None, collection_name: str = "docmind"):
    """একটা document লোড → chunk → index করে।"""
    source_name = source_name or file_path.split("/")[-1]
    text = load_document(file_path)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    collection = get_collection(collection_name)
    num_added = index_chunks(chunks, collection, source_name=source_name)
    print(f"✅ {source_name}: {num_added} chunks indexed")
    return collection


def ask(query: str, collection, top_k: int = 3):
    """Query নিয়ে retrieval + generation করে, answer ও source metadata রিটার্ন করে।"""
    docs, metadatas = search(query, collection, top_k=top_k)
    answer = generate_answer(query, docs)
    return {
        "answer": answer,
        "sources": metadatas,
        "retrieved_chunks": docs,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <path_to_pdf_or_txt> <your_question>")
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    collection = build_index(file_path)
    result = ask(question, collection)

    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Sources ---")
    for m in result["sources"]:
        print(m)

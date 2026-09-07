"""
Step 3 & 4: Embedding + Vector Store
সম্পূর্ণ local — কোনো API key লাগে না।
Model: all-MiniLM-L6-v2 (sentence-transformers) — প্রথমবার চালানোর সময়
       একবার ইন্টারনেট থেকে ডাউনলোড হবে (~90MB), তারপর অফলাইনে কাজ করে।
"""
import chromadb
from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection(collection_name: str = "docmind", persist_dir: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)
    return collection


def index_chunks(chunks: list[str], collection, source_name: str = "document"):
    """Chunks কে embed করে vector store এ যোগ করে।"""
    model = get_model()
    embeddings = model.encode(chunks).tolist()
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query: str, collection, top_k: int = 5):
    """Query এর সাথে সবচেয়ে relevant chunks খুঁজে বের করে।"""
    model = get_model()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


if __name__ == "__main__":
    # sanity test
    col = get_collection("test_collection")
    index_chunks(["বিড়াল একটি প্রাণী।", "পাইথন একটি প্রোগ্রামিং ভাষা।"], col, source_name="test")
    docs, metas = search("প্রোগ্রামিং ভাষা কি?", col, top_k=1)
    print(docs)

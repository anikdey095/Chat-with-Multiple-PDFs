"""
Step 2: Chunking
বড় text কে ছোট ছোট, overlapping অংশে ভাগ করে —
যাতে embedding model ও LLM এর জন্য manageable হয়।
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Word-count ভিত্তিক chunking। chunk_size ও overlap word-এ হিসেব করা হয়।

    chunk_size=300, overlap=50 মানে: প্রতিটা chunk ৩০০ শব্দের,
    পরের chunk আগেরটার শেষ ৫০ শব্দ থেকে শুরু হবে (context না হারানোর জন্য)।
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    sample = "এটা একটা " * 1000  # dummy long text দিয়ে টেস্ট
    result = chunk_text(sample, chunk_size=300, overlap=50)
    print(f"Total chunks: {len(result)}")
    print(f"First chunk word count: {len(result[0].split())}")

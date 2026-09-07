"""
Step 1: Document Loading
PDF থেকে raw text বের করে আনে।
"""
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """PDF ফাইল থেকে সব পাতার text একসাথে জোড়া দিয়ে রিটার্ন করে।"""
    reader = PdfReader(path)
    full_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        full_text.append(text)
    return "\n".join(full_text)


def load_txt(path: str) -> str:
    """Plain .txt ফাইল লোড করে।"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(path: str) -> str:
    """Extension দেখে সঠিক loader বেছে নেয়।"""
    if path.lower().endswith(".pdf"):
        return load_pdf(path)
    elif path.lower().endswith(".txt"):
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    # ছোট sanity test — data/ ফোল্ডারে একটা sample.pdf বা sample.txt রাখুন
    import sys
    if len(sys.argv) > 1:
        text = load_document(sys.argv[1])
        print(f"Loaded {len(text)} characters")
        print(text[:300])
    else:
        print("Usage: python loader.py <path_to_file>")

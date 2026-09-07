"""
Step 5: Generation
সম্পূর্ণ local LLM — Ollama দিয়ে। কোনো API key/cost লাগে না।

প্রথমে Ollama ইন্সটল করতে হবে (একবারই):
  1. https://ollama.com থেকে Ollama ইন্সটল করুন (Windows/Mac/Linux সব সাপোর্ট করে)
  2. Terminal এ চালান:  ollama pull llama3.2:1b
     (এটা ছোট, দ্রুত মডেল — কম RAM এও চলে। শক্তিশালী PC হলে llama3.2:3b বা
      phi3 ব্যবহার করতে পারেন — শুধু নিচের MODEL_NAME বদলে দিন)
  3. ollama serve ব্যাকগ্রাউন্ডে চলতে হবে (সাধারণত install করলেই auto-start হয়)
"""
import ollama

MODEL_NAME = "llama3.2:1b"

PROMPT_TEMPLATE = """তুমি একজন সহায়ক assistant। নিচের context ব্যবহার করে প্রশ্নের উত্তর দাও।
Context এ উত্তর না থাকলে স্পষ্টভাবে বলো "এই তথ্য document এ নেই" — কখনো নিজে থেকে বানিয়ে বলবে না।

Context:
{context}

প্রশ্ন: {query}

উত্তর (কোন অংশ থেকে উত্তর নিয়েছ সংক্ষেপে উল্লেখ করো):"""


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    # sanity test — আগে "ollama pull llama3.2:1b" চালিয়ে নিন
    fake_context = ["ঢাকা বাংলাদেশের রাজধানী। এখানে প্রায় ২ কোটি মানুষ বাস করে।"]
    answer = generate_answer("বাংলাদেশের রাজধানী কোথায়?", fake_context)
    print(answer)

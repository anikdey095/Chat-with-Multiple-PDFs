# DocMind — Setup Guide (100% Local, No API Key)

## Stack
- **LLM**: Ollama (local, free) — `llama3.2:1b`
- **Embedding**: `sentence-transformers` (`all-MiniLM-L6-v2`, local, free)
- **Vector DB**: Chroma (local, free)
- **UI**: Streamlit

কোনো OpenAI/Claude API key লাগবে না। ইন্টারনেট শুধু প্রথমবার model download করার জন্য লাগবে, তারপর সব offline চলবে।

---

## Setup (এই ধাপগুলো নিজের কম্পিউটারে করুন)

### 1. Ollama Install করুন
- https://ollama.com এ যান, নিজের OS (Windows/Mac/Linux) অনুযায়ী ইন্সটল করুন
- Install হলে terminal এ চেক করুন:
  ```bash
  ollama --version
  ```

### 2. একটা ছোট model download করুন
```bash
ollama pull llama3.2:1b
```
এটা মোটামুটি ১.৩GB, কম RAM (৪-৮GB) এও চলে। যদি আপনার PC শক্তিশালী হয় (16GB+ RAM), `src/generator.py` তে `MODEL_NAME` বদলে `llama3.2:3b` বা `phi3` দিতে পারেন — উত্তর quality আরও ভালো হবে।

### 3. Python environment বানান
```bash
cd docmind
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Sample document রাখুন
`data/` ফোল্ডারে একটা ছোট PDF বা TXT ফাইল রাখুন (২-১০ পাতা দিয়ে শুরু করুন)।

### 5. Terminal দিয়ে টেস্ট করুন
```bash
cd src
python pipeline.py ../data/your_file.pdf "আপনার প্রশ্ন এখানে লিখুন"
```

### 6. UI চালান
```bash
streamlit run app.py
```
Browser এ `http://localhost:8501` খুলবে।

---

## যদি সমস্যা হয়

| সমস্যা | সমাধান |
|---|---|
| `ollama.chat` connection error | `ollama serve` আলাদা terminal এ চালান, তারপর আবার try করুন |
| Embedding model download হচ্ছে না | প্রথমবার internet লাগবে (~90MB), পরে offline কাজ করবে |
| PDF থেকে text আসছে না | Scanned/image PDF হলে এই setup কাজ করবে না, OCR লাগবে (advanced topic, পরে) |
| উত্তর ভালো আসছে না | model ছোট বলে হতে পারে — `llama3.2:3b` try করুন, অথবা chunk_size/top_k tune করুন |

---

## এরপর কি করবেন (Day 5-7 অনুযায়ী)
1. `eval/` ফোল্ডারে ১৫-২০টা প্রশ্ন-উত্তরের test set বানান
2. RAGAS দিয়ে evaluate করুন (এই ধাপে সাহায্য দরকার হলে বলবেন, আমি evaluate.py লিখে দেব)
3. chunk_size, top_k নিয়ে experiment করে score compare করুন

কাজ করছে কিনা, বা কোনো error পেলে জানাবেন — একসাথে debug করব।

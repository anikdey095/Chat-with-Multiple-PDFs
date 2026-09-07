*"""
DocMind — Streamlit UI
চালানো: streamlit run app.py
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from pipeline import build_index, ask

st.set_page_config(page_title="DocMind", page_icon="📄")
st.title("📄 DocMind — নিজের Document দিয়ে প্রশ্ন করুন")
st.caption("সম্পূর্ণ local ও free — কোনো API key লাগে না (Ollama ব্যবহার করছে)")

if "collection" not in st.session_state:
    st.session_state.collection = None

uploaded_file = st.file_uploader("PDF বা TXT ফাইল আপলোড করুন", type=["pdf", "txt"])

if uploaded_file and st.button("Index করুন"):
    with st.spinner("Document পড়া ও index করা হচ্ছে..."):
        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.session_state.collection = build_index(tmp_path, source_name=uploaded_file.name)
    st.success("Index তৈরি হয়ে গেছে! এখন প্রশ্ন করতে পারেন।")

if st.session_state.collection:
    query = st.text_input("আপনার প্রশ্ন লিখুন")
    if st.button("জিজ্ঞাসা করুন") and query:
        with st.spinner("উত্তর খোঁজা হচ্ছে..."):
            result = ask(query, st.session_state.collection)
        st.markdown("### উত্তর")
        st.write(result["answer"])
        with st.expander("কোন অংশ থেকে উত্তর নেওয়া হয়েছে দেখুন (Source)"):
            for i, (chunk, meta) in enumerate(zip(result["retrieved_chunks"], result["sources"])):
                st.markdown(f"**Chunk {i+1}** — `{meta}`")
                st.text(chunk[:400] + "...")/*

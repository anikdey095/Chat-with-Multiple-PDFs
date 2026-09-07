import os
import html
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
from huggingface_hub import InferenceClient
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter
from html_template import css, bot_template, user_template


class HuggingFaceChat(BaseChatModel):
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    api_token: str = Field(default="")
    temperature: float = 0.1
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        hf_messages = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else ("assistant" if isinstance(m, AIMessage) else "system")
            hf_messages.append({"role": role, "content": m.content})

        token = self.api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        client = InferenceClient(token=token)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=hf_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])


def init_environment():
    load_dotenv()
    # Normalize environment variable names for compatibility
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = os.environ.get("openai_api_key")
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get("hugginfface_api_key"):
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("hugginfface_api_key")
        elif os.environ.get("HUGGINGFACE_API_KEY"):
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACE_API_KEY")
    if not os.environ.get("HF_TOKEN") and os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.error(f"Error reading {getattr(pdf, 'name', 'PDF')}: {str(e)}")
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, model_choice="Hugging Face (Llama-3.2)"):
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if model_choice.startswith("OpenAI") and openai_key:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    else:
        llm = HuggingFaceChat(api_token=hf_token or "")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    if not user_question:
        return

    conversation = st.session_state.get("conversation")
    if conversation is None:
        st.warning("Process at least one PDF before asking a question.")
        return

    try:
        with st.spinner("Thinking..."):
            response = conversation.invoke({"question": user_question})

        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "bot", "content": response.get("answer", "")})
        st.rerun()
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "credit_balance_exhausted" in error_msg:
            st.error("OpenAI credit quota exhausted. Switch to 'Hugging Face' in the sidebar to use your Hugging Face API key.")
        else:
            st.error(f"Error generating response: {error_msg}")


def render_chat_history():
    if "messages" in st.session_state and st.session_state.messages:
        for message in st.session_state.messages:
            safe_content = html.escape(message["content"]).replace("\n", "<br>")
            if message["role"] == "user":
                st.write(user_template.replace("{{MSG}}", safe_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", safe_content), unsafe_allow_html=True)


def main():
    init_environment()

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header("Chat with multiple PDF documents :books:")

    render_chat_history()

    user_question = st.chat_input("Ask a question about your PDFs:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Model Configuration")
        model_choice = st.selectbox(
            "Choose LLM Model:",
            options=["Hugging Face (Llama-3.2)", "OpenAI (GPT-3.5)"],
            index=0,
            help="Hugging Face uses your Hugging Face API key; OpenAI requires active OpenAI account credits.",
        )

        st.subheader("Upload your PDF documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", type="pdf", accept_multiple_files=True
        )
        if st.button("Process"):
            if not uploaded_files:
                st.warning("Upload at least one PDF first.")
                return

            with st.spinner("Processing documents..."):
                try:
                    # Extract text from uploaded PDF files
                    raw_text = get_pdf_text(uploaded_files)
                    if not raw_text.strip():
                        st.warning("The uploaded PDFs contain no extractable text.")
                        return

                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = create_vectorstore(text_chunks)

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, model_choice=model_choice)
                    st.success("PDFs processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred while processing PDFs: {str(e)}")

        if st.session_state.messages:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
import os
import sys
import html
import warnings
import streamlit as st
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# PDF Reader: pypdf is modern standard, PyPDF2 as fallback
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

# Standalone CharacterTextSplitter: fast, zero-hanging, 100% compatible
class CharacterTextSplitter:
    """Fast, dependency-free CharacterTextSplitter compatible with LangChain interface."""
    def __init__(
        self,
        separator: str = "\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function=len,
    ):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        splits = text.split(self.separator)
        chunks = []
        current_chunk = []
        current_len = 0

        for s in splits:
            s_len = self.length_function(s)
            if current_len + s_len > self.chunk_size and current_chunk:
                doc = self.separator.join(current_chunk).strip()
                if doc:
                    chunks.append(doc)
                while current_chunk and current_len > self.chunk_overlap:
                    removed = current_chunk.pop(0)
                    current_len -= self.length_function(removed) + len(self.separator)
            current_chunk.append(s)
            current_len += s_len + len(self.separator)

        if current_chunk:
            doc = self.separator.join(current_chunk).strip()
            if doc:
                chunks.append(doc)

        return chunks


# LangChain core primitives
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import Field
from huggingface_hub import InferenceClient

# Import HTML/CSS templates (supports both naming conventions)
try:
    from html_template import css, bot_template, user_template
except ImportError:
    from htmlTemplates import css, bot_template, user_template


class HuggingFaceChat(BaseChatModel):
    """Custom LangChain chat model utilizing Hugging Face Inference API."""
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
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "user"
            hf_messages.append({"role": role, "content": str(m.content)})

        token = (
            self.api_token
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("hugginfface_api_key")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HF_TOKEN")
        )
        if not token:
            raise ValueError(
                "Hugging Face API token not found! Please set HUGGINGFACEHUB_API_TOKEN in your .env file."
            )

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


class ConversationBufferMemory:
    """Fast, in-memory conversation buffer that avoids hanging on legacy package imports."""
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = InMemoryChatMessageHistory()

    @property
    def messages(self):
        return self.chat_memory.messages

    def save_context(self, inputs: dict, outputs: dict):
        q = inputs.get("question") or inputs.get("input") or (list(inputs.values())[0] if inputs else "")
        a = outputs.get("answer") or outputs.get("output") or (list(outputs.values())[0] if outputs else "")
        self.chat_memory.add_message(HumanMessage(content=str(q)))
        self.chat_memory.add_message(AIMessage(content=str(a)))

    def load_memory_variables(self, inputs: dict = None):
        if self.return_messages:
            return {self.memory_key: self.chat_memory.messages}
        buffer_str = "\n".join([f"{m.type}: {m.content}" for m in self.chat_memory.messages])
        return {self.memory_key: buffer_str}

    def clear(self):
        self.chat_memory.clear()


class ConversationalRetrievalChain:
    """High-performance Conversational Retrieval QA Chain with zero hanging imports."""
    def __init__(self, llm, retriever, memory=None):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory if memory is not None else ConversationBufferMemory()

    @classmethod
    def from_llm(cls, llm, retriever, memory=None):
        return cls(llm=llm, retriever=retriever, memory=memory)

    def invoke(self, inputs: dict):
        user_question = inputs.get("question") or inputs.get("input") or ""

        # Retrieve top relevant context chunks
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(user_question)
            except AttributeError:
                docs = self.retriever.get_relevant_documents(user_question)

        context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant context found in documents."

        # Collect past history messages
        history_msgs = []
        if self.memory and hasattr(self.memory, "chat_memory"):
            history_msgs = self.memory.chat_memory.messages
        elif self.memory and hasattr(self.memory, "messages"):
            history_msgs = self.memory.messages

        system_instruction = (
            "You are a helpful and knowledgeable AI assistant answering questions based on the provided PDF documents.\n"
            "Use the following pieces of retrieved context to answer the question thoroughly and accurately.\n"
            "If the answer cannot be found in the context, politely state that the information is not present in the documents.\n\n"
            f"--- RELEVANT CONTEXT ---\n{context}\n------------------------"
        )

        prompt_messages = [SystemMessage(content=system_instruction)]
        # Add up to 6 previous history messages for context
        prompt_messages.extend(history_msgs[-6:])
        prompt_messages.append(HumanMessage(content=user_question))

        # Generate answer from LLM
        res = self.llm.invoke(prompt_messages)
        answer = res.content if hasattr(res, "content") else str(res)

        # Save turn to conversation memory
        if self.memory:
            self.memory.save_context({"question": user_question}, {"answer": answer})

        return {
            "answer": answer,
            "chat_history": self.memory.chat_memory.messages if (self.memory and hasattr(self.memory, "chat_memory")) else [],
            "source_documents": docs,
        }

    def __call__(self, inputs: dict):
        return self.invoke(inputs)


def init_environment():
    """Load and normalize environment variables."""
    load_dotenv()
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
    """Extract text from uploaded PDF files."""
    if not pdf_docs:
        return ""
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
    """Split raw text into manageable chunks with overlap."""
    if not raw_text:
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """Cache the embeddings model to prevent re-instantiation across runs."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vectorstore(text_chunks):
    """Create a FAISS vector store from text chunks."""
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        from langchain.vectorstores import FAISS

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Alias for tutorial code compatibility
get_vectorstore = create_vectorstore


def get_conversation_chain(vectorstore, model_choice="Hugging Face (Llama-3.2)"):
    """Create conversational retrieval chain with selected LLM backend."""
    hf_token = (
        os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("hugginfface_api_key")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HF_TOKEN")
    )
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")

    if model_choice.startswith("OpenAI") and openai_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI
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
    """Handle user prompt, invoke conversational chain, and update chat history."""
    if not user_question:
        return

    conversation = st.session_state.get("conversation")
    if conversation is None:
        st.warning("Please upload and process at least one PDF before asking a question.")
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
            st.error("OpenAI credit quota exhausted. Switch to 'Hugging Face (Llama-3.2)' in the sidebar to use your free Hugging Face API key.")
        elif "401" in error_msg or "Unauthorized" in error_msg or "token" in error_msg.lower():
            st.error("Hugging Face authorization error. Please check your HUGGINGFACEHUB_API_TOKEN in .env.")
        else:
            st.error(f"Error generating response: {error_msg}")


def render_chat_history():
    """Render conversation message history using custom HTML/CSS templates."""
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
            help="Hugging Face uses your Hugging Face API token (free); OpenAI requires active OpenAI account credits.",
        )

        st.subheader("Upload your PDF documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", type="pdf", accept_multiple_files=True
        )
        if st.button("Process"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file before processing.")
                return

            with st.spinner("Processing documents..."):
                try:
                    # Extract text from uploaded PDF files
                    raw_text = get_pdf_text(uploaded_files)
                    if not raw_text.strip():
                        st.warning("The uploaded PDFs contain no extractable text. Please ensure they are digital PDFs with selectable text (not scanned images).")
                        return

                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.warning("Could not extract any chunks from the uploaded documents.")
                        return

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
                if st.session_state.conversation and hasattr(st.session_state.conversation, "memory") and st.session_state.conversation.memory:
                    st.session_state.conversation.memory.clear()
                st.rerun()


if __name__ == "__main__":
    main()
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()
sys.modules["sqlite3"] = pysqlite3
# Persistent directory for ChromaDB
PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)

os.environ["HUGGINGFACE_KEY"] = st.secrets["HUGGINGFACE_KEY"]
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar for settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Groq API key", type="password")
    session_id = st.text_input("Session ID", value="default_session")

# Title
st.title("📄 Chat with your PDF")
st.write("Upload a PDF and ask questions about its content.")

if api_key:
    llm = ChatGroq(api_key=api_key, model_name="llama3-8b-8192")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        # Load and split document
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temppdf)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name="pdf_collection"
        )
        retriever = vectorstore.as_retriever()

        # Prompts
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don’t know the answer, say so. "
            "Use three sentences maximum and try to be concise and to the point. "
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.success(response["answer"])
    else:
        st.info("Upload a PDF file to begin.")
else:
    st.info("Enter your Groq API key in the sidebar to begin.")

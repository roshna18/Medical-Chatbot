import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import PyPDF2

# Configuration
st.set_page_config(page_title="🩺 Medical Assistant Chatbot", page_icon="🩺", layout="wide")

# Constants
PDF_PATH = "C:/Users/Manju/OneDrive/Desktop/Projects/Medical-Chatbot/Data/RefBook.pdf"
FAISS_INDEX_PATH = "C:/Users/Manju/OneDrive/Desktop/Projects/Medical-Chatbot/research/faiss_index"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA7rF3n6D24zICDBH80sYfiEyhyMFDo82M"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1976d2;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar Controls
with st.sidebar:
    st.header("🔧 Configuration")
    st.info(f"📄 Using PDF: RefBook.pdf")

    depth_level = st.slider("📊 Response Depth", 1, 5, 3)
    reference_adherence = st.slider("📚 Reference Adherence", 1, 5, 3)
    k_documents = st.slider("🔍 Source Documents", 2, 10, 4)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# Text Extraction from PDF
@st.cache_data
def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Vectorstore Setup
@st.cache_resource
def setup_vectorstore():
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ PDF not found at {PDF_PATH}")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        raw_text = extract_text_from_pdf(PDF_PATH)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = FAISS.from_documents(documents, embeddings)
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore

# QA Chain Creation
def create_qa_chain(vectorstore, depth, adherence, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    depth_map = {
        1: "Provide a brief, concise answer.",
        2: "Provide a clear, straightforward answer.",
        3: "Provide a detailed explanation with examples.",
        4: "Provide a comprehensive answer with background information.",
        5: "Provide an in-depth, thorough explanation with multiple perspectives."
    }

    adherence_map = {
        1: "You can use general medical knowledge along with the provided context.",
        2: "Primarily use the provided context, but can supplement with general knowledge.",
        3: "Balance between provided context and general medical knowledge.",
        4: "Strongly prioritize the provided context over general knowledge.",
        5: "Strictly use only the information from the provided context."
    }

    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=f"""
You are a helpful medical assistant. {adherence_map[adherence]}

{depth_map[depth]}

Context:
{{context}}

Chat History:
{{chat_history}}

Question:
{{question}}

Note: This is for educational purposes only. Always consult healthcare professionals.

Answer:"""
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# Main UI
st.markdown("<h1 class='main-header'>🩺 Medical Assistant Chatbot</h1>", unsafe_allow_html=True)

with st.spinner("🔄 Loading medical reference and preparing system..."):
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = setup_vectorstore()

    if st.session_state.vectorstore:
        st.session_state.qa_chain = create_qa_chain(
            st.session_state.vectorstore,
            depth_level,
            reference_adherence,
            k_documents
        )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"question": prompt})
                    answer = response["answer"] + "\n\n---\n⚠️ *This information is for educational purposes only. Please consult healthcare professionals for medical advice.*"
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err = f"❌ Error generating response: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import PyPDF2

# Set your Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA7rF3n6D24zICDBH80sYfiEyhyMFDo82M"

# === Step 1: Extract Text from PDF ===
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

# === Step 2: Prepare documents ===
pdf_path = "C:/Users/Manju/OneDrive/Desktop/Projects/Medical-Chatbot/Data/RefBook.pdf"
faiss_index_path = "C:/Users/Manju/OneDrive/Desktop/Projects/Medical-Chatbot/research/faiss_index"

raw_text = extract_text_from_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(raw_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# === Step 3: Embed and Load Vectorstore ===
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(faiss_index_path):
    print("📦 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(faiss_index_path)
else:
    print("📂 Loading existing FAISS vectorstore...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# === Step 4: Set up Gemini LLM and Prompt ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful medical assistant. Use the following context to answer the question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

# === Step 5: Answer Query with Summarization If Needed ===
def answer_query(query, summary_trigger_tokens=150):
    result = qa_chain.invoke({"query": query})
    answer = result['result']

    if len(answer.split()) > summary_trigger_tokens:
        summary_prompt = f"Summarize the following medical answer into a concise paragraph:\n\n{answer}"
        answer = llm.invoke(summary_prompt)

    print(f"\n💡 Answer: {answer}\n")

# === Step 6: Interactive CLI ===
if __name__ == "__main__":
    while True:
        user_input = input("🩺 Ask your medical question (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer_query(user_input)
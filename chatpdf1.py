import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say "answer is not available in the context".
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)  # Fixed model & temp
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

st.set_page_config(page_title="📄 Chat with PDFs", layout="wide")

st.sidebar.title("📂 PDF Chat Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["💬 Chat", "📂 Upload PDFs"])

with tab2:
    st.header("📂 Upload and Process PDFs", divider="rainbow")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
    if st.button("🚀 Submit & Process"):
        if pdf_docs:
            with st.spinner("Extracting and processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.toast("✅ Processing complete!", icon="🎉")
        else:
            st.warning("Please upload at least one PDF.")

with tab1:
    st.header("💬 Ask Questions", divider="blue")
    user_question = st.text_input("Type your question:")
    if user_question:
        with st.spinner("Thinking..."):
            answer = user_input(user_question)
            st.session_state.history.append({"question": user_question, "answer": answer})
            st.toast("💡 Answer generated!", icon="🤖")
    
    for chat in reversed(st.session_state.history):
        st.markdown(f"**🧑‍💻 You:** {chat['question']}")
        st.markdown(f"**🤖 Bot:** {chat['answer']}")
        st.divider()


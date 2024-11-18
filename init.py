import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

@st.cache_resource
def init_rag():
    # Folder tempat menyimpan dokumen .txt
    txt_folder_path = "./Data/"
    all_txt_paths = glob.glob(os.path.join(txt_folder_path, "*.txt"))
    
    # Memuat dokumen dan metadata
    documents = []
    for txt_path in all_txt_paths:
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            metadata = {
                "title": lines[0].strip().replace("Judul: ", ""),
                "author": lines[1].strip().replace("Penulis: ", ""),
                "type": lines[2].strip().replace("Jenis: ", ""),
                "source": txt_path
            }
            content = "".join(lines[4:])  # Isi dokumen mulai dari baris ke-5
            documents.append({"content": content, "metadata": metadata})

    # Memecah dokumen menjadi chunk kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            split_documents.append({
                "content": chunk,
                "metadata": doc["metadata"]
            })
    
    # Inisialisasi embeddings dan model LLM dengan Google Gemini API
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    # Membuat FAISS vector database
    vector_db = FAISS.from_documents(
        [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in split_documents],
        embeddings
    )

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever
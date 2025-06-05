import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

pdfs_directory = 'pdfs/'
FAISS_DB_PATH = 'vectorstore/db_faiss'
ollama_model_name = "deepseek-r1:7b"

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    path = os.path.join(pdfs_directory, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def create_chunks(documents, source):
    for doc in documents:
        doc.metadata["source"] = source
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def get_embedding_model(model_name=ollama_model_name):
    return OllamaEmbeddings(model=model_name)

def update_or_create_faiss(text_chunks, embedding_model):
    if os.path.exists(FAISS_DB_PATH):
        db = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(text_chunks)
    else:
        db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(FAISS_DB_PATH)
    return db

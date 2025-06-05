import os
import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model
from vector_database import upload_pdf, load_pdf, create_chunks, get_embedding_model, update_or_create_faiss

FAISS_DB_PATH = "vectorstore/db_faiss"

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
user_query = st.text_area("Enter your prompt:  ", height=150, placeholder="Ask Anything!")
ask_question = st.button("Ask AI lawyer")

if ask_question:
    if uploaded_file:
        st.chat_message("User: ").write(user_query)

        # Save and process uploaded PDF
        upload_pdf(uploaded_file)
        file_path = os.path.join("pdfs", uploaded_file.name)
        documents = load_pdf(file_path)
        chunks = create_chunks(documents, source=uploaded_file.name)
        embedding_model = get_embedding_model()
        update_or_create_faiss(chunks, embedding_model)

        # Retrieve from updated FAISS DB
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(retrieved_docs, llm_model, user_query)
        st.chat_message("AI Lawyer: ").write(response)
    else:
        st.error("Please upload a PDF first!")

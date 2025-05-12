from rag_pipeline import answer_query, retrieve_docs, llm_model
# STep 1 : Setup PDF upload functionality
import streamlit as st

uploaded_file = st.file_uploader("Upload PDF",
                                 type = "pdf",
                                 accept_multiple_files= False)

#Step 2: ChatBot Skeleton( Q & A)
user_query = st.text_area("Enter your prompt:  ",height=150,placeholder="Ask Anything!")

ask_question = st.button("Ask AI lawyer")

if (ask_question):

    if(uploaded_file):
        st.chat_message("User: ").write(user_query)

        # RAG pipeline
        retrieved_docs=retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        
        #fixed_response = "HI this is fixed response"
        st.chat_message("AI Lawyer: ").write(response)
    else:
        st.error("Kindly upload a valid pdf file first! ")


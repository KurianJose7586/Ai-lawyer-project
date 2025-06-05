from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from vector_database import get_embedding_model

import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=api_key)

FAISS_DB_PATH = "vectorstore/db_faiss"
embedding_model = get_embedding_model()

def retrieve_docs(query):
    db = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know." Only use the given context.

Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

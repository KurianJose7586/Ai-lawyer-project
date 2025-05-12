# 🧠 AI Lawyer – Legal Document Q&A with RAG

This project is an AI-powered chatbot that can answer questions based on uploaded legal PDFs. It uses a **Retrieval-Augmented Generation (RAG)** pipeline with a vector database (FAISS), PDF document parsing, and a powerful LLM (DeepSeek via Groq) to provide accurate, context-based responses.

## 🚀 Features

- Upload a PDF document (e.g., legal texts).
- Ask natural language questions about its content.
- Receive contextually accurate answers from an LLM.
- Uses FAISS for vector search and LangChain for LLM orchestration.
- Streamlit frontend for easy interaction.

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **LLM Backend**: DeepSeek R1 via Groq API  
- **Vector Store**: FAISS  
- **Embeddings**: Ollama DeepSeek R1  
- **PDF Parsing**: PDFPlumber via LangChain  
- **Environment Management**: `python-dotenv`

## 📁 Project Structure

```
.
├── frontend.py          # Streamlit interface
├── rag_pipeline.py      # Core RAG logic (LLM, prompt, retrieval)
├── vector_database.py   # PDF processing, chunking, vector storage
├── pdfs/                # Folder for uploaded PDFs
└── vectorstore/         # Saved FAISS database
```

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-lawyer.git
   cd ai-lawyer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Add your PDF document**

   Place a legal PDF document into the `pdfs/` directory or upload via the Streamlit app.

5. **Run the app**
   ```bash
   streamlit run frontend.py
   ```

## 🧪 Example Prompt

> "If a government forbids the right to assemble peacefully which articles are violated and why?"

The system retrieves relevant sections from the uploaded document and generates a precise, grounded answer.

## 📌 Notes

- Ensure the Ollama embedding model (`deepseek-r1:1.5b`) is available in your environment.
- You can switch to other LLMs and embedding models with minor adjustments.


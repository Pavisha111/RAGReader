# RAGReader
QD Bot is an AI-powered assistant built to interact with PDF documents. It allows users to upload PDFs, convert them into markdown format, and query them for relevant information using an LLM-based search system.
Features

    PDF to Markdown Conversion: Automatically converts PDFs to markdown.
    Interactive Query System: Ask questions and get context-based, concise answers from the document.
    Memory Buffer: Tracks chat history for context-aware conversations.

Key Technologies

    Streamlit: Front-end interface.
    Ollama LLM (Mistral): For generating responses based on document content.
    Qdrant: Vector store for efficient document indexing and search.

How to Use

    Upload a PDF document.
    Ask questions related to the document.
    Get relevant, accurate answers.
    
Installation

git clone https://github.com/your-repo/qdbot.git
cd qdbot
pip install -r requirements.txt
streamlit run app.py    

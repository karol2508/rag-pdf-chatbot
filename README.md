# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions 
based on the content of PDF documents.

## How it works

1. Load a PDF document
2. Split it into chunks and create embeddings
3. Store embeddings in a FAISS vector database
4. Ask questions — the bot retrieves relevant context and answers

## Tech Stack

- Python
- LangChain
- OpenAI API (GPT-3.5-turbo)
- FAISS (vector database)
- PyPDF

## Installation

pip install -r requirements.txt

## Usage

1. Add a PDF file to the `docs/` folder
2. Create a `.env` file with your OpenAI API key:
OPENAI_API_KEY=sk-your-key-here
3. Run:
python rag_chatbot.py

## Skills demonstrated

- RAG (Retrieval Augmented Generation)
- Vector databases (FAISS)
- Document processing and chunking
- LangChain framework
- OpenAI API integration

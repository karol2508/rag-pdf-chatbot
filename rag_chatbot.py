from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

def load_pdf(path):
    print(f"Loading PDF: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages

def create_vectorstore(pages):
    print("Creating vector store...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store created and saved!")
    return vectorstore

def get_relevant_context(vectorstore, question, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def ask_question(llm, vectorstore, question):
    context = get_relevant_context(vectorstore, question)
    messages = [
        SystemMessage(content=f"""You are a helpful assistant that answers questions 
        based on the provided document context. Always base your answers on the context.
        If the answer is not in the context, say so clearly.
        
        Context:
        {context}"""),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content

def main():
    print("RAG PDF Chatbot")
    print("==================")
    
    pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in docs/ folder")
        print("Add a PDF file to the docs/ folder and try again")
        return
    
    print(f"\nAvailable PDFs: {pdf_files}")
    pdf_path = f"docs/{pdf_files[0]}"
    
    pages = load_pdf(pdf_path)
    vectorstore = create_vectorstore(pages)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    print("\nReady, ask questions about your PDF.")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        print("\nThinking...")
        answer = ask_question(llm, vectorstore, question)
        print(f"\nBot: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
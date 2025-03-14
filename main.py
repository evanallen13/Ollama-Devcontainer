import os
from src.llama import llama
from src.database import database
from src.document_loader import document_loader

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = "gemma3:1b" # Find available models here https://ollama.com/library

if __name__ == "__main__":
    
    doc_loader = document_loader()
    documents = doc_loader.load_documents()
    sliced_documents = doc_loader.split_documents(documents)
    
    # db = database("database")
    # embeddings = db.get_embedding_function()
    

    # llama = llama(OLLAMA_HOST, MODEL)

    # prompt = input("Input prompt: ")
    # llama.generate_response(prompt)
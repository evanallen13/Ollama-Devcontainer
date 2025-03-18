import os
from src.llama import llama
from src.database import database
from src.document_loader import document_loader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = "gemma3:1b" # Find available models here https://ollama.com/library

if __name__ == "__main__":
    
    doc_loader = document_loader()
    documents = doc_loader.load_documents()
    sliced_documents = doc_loader.split_documents(documents)
    
    db = database("database")
    db.add_to_chroma(sliced_documents)
    
    os.system("clear")
    # query_text = "What happens if you land on a property in Monopoly that you can't afford?"
    
   
    

    llama = llama(OLLAMA_HOST, MODEL)

    # prompt = input("Input prompt: ")
    prompt = "What happens if you land on a property in Monopoly that you can't afford?"
    results = db.similarity_search_with_score(prompt, k=3)
    results_with_content = ""
    
    for i in results:
        results_with_content += i[0].page_content + " "
    
    print(results[0][0].page_content)
    llama.generate_response(f"""
                            You are a expert on the rules of the games Monopoly and Ticket to Ride. 
                            Using these documenets ${results_with_content} and the prompt ${prompt}
                            """)
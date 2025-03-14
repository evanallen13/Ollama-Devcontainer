from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class document_loader: 
    
    def load_documents(self, DATA_PATH="data"):
        document_loader = PyPDFDirectoryLoader(DATA_PATH)

        return document_loader.load()
    
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)
    

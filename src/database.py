from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"

class database: 
    
    def __init__(self, db_name):
        self.db_name = db_name
        
    def get_embedding_function(self):
        # embeddings = BedrockEmbeddings(
        #     credentials_profile_name="default", region_name="us-east-1"
        # )
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings



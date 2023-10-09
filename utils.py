import pinecone
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Set up your OpenAI API key
load_dotenv()
pinecone.init(api_key="4176760b-9c97-40d4-8d61-58f2b55d3518", environment="gcp-starter")

# Define the name for your Pinecone index
index_name = 'fir'

def find_match(input,model):
    embeddings = OpenAIEmbeddings(model_name=model)
    embedding = embeddings.embed_query(input)
    pinecone_index = pinecone.Index(index_name)
    response = pinecone_index.query(
        vector=embedding,
        top_k=2,
        # include_values=True,
        includeMetadata=True
    )
    return response['matches'][0]['metadata']['text'] + "\n" + response['matches'][1]['metadata']['text']



import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load .env
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # For text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENVIRONMENT"))
    )

index = pc.Index(index_name)

# Function to create embeddings
def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Function to upsert document
def upsert_document(doc_id: str, text: str):
    embedding = get_embedding(text)
    index.upsert(vectors=[{"id": doc_id, "values": embedding, "metadata": {"text": text}}])
    return {"status": "inserted", "id": doc_id}

# Function to query similar documents
def query_similar(text: str, top_k: int = 3):
    embedding = get_embedding(text)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results
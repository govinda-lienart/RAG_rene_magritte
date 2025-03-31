import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "medium-blogs-embedding-index")

# Connect to your specific index
index = pc.Index(index_name)

# Delete all vectors from the index
index.delete(delete_all=True)

print(f"✅ All vectors deleted from the index '{index_name}'. Index itself remains.")

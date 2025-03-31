import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()

# === Read API keys and index name ===
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# === Initialize clients ===
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(index_name)

# === Define a query ===
query_text = "What is a vector database?"

# === Create embedding for the query ===
query_response = client.embeddings.create(
    input=[query_text],
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

# === Query Pinecone ===
search_result = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# === Display top results ===
print(f"\n🔍 Top {len(search_result['matches'])} most relevant chunks for query: \"{query_text}\"")
print("==============================================")

for match in search_result['matches']:
    score = match['score']
    metadata = match['metadata']
    chunk_num = metadata.get('chunk', 'N/A')
    source = metadata.get('source', 'N/A')
    text = metadata.get('text', '[No text stored — add it to metadata during upsert if needed]')

    print(f"\n📄 Chunk #{chunk_num} (Score: {score:.2f})")
    print(f"📁 Source: {source}")
    print("📝 Text preview:")
    print(text[:300] + ("..." if len(text) > 300 else ""))
    print("—" * 50)

print("\n✅ Retrieval complete.")
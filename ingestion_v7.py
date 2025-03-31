import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Step 1: Read API keys and index name from environment
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# ✅ Create a unique run ID based on the current timestamp
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# === Init Clients ===
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(index_name)

# === Load Document ===
file_path = "/Users/govinda-dashugolienart/Library/CloudStorage/GoogleDrive-govinda.lienart@three-monkeys.org/My Drive/TMWC - Govinda /TMWC - Govinda /Data Science/Environments/Pycharm/intro-to-vector-dbs/downloaded_file.text"
loader = TextLoader(file_path)
docs = loader.load()

# === Split into Chunks ===
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks for run_id: {run_id}")

# === Embed & Upsert Each Chunk (corrected with text metadata) ===
for i, chunk in enumerate(chunks):
    text = chunk.page_content
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    vector_id = f"{run_id}-doc-{i:03}"
    index.upsert([{
        "id": vector_id,
        "values": embedding,
        "metadata": {
            "chunk": i,
            "source": os.path.basename(file_path),
            "run_id": run_id,
            "text": text  # ✅ explicitly adding text content
        }
    }])

    print(f"✅ Upserted chunk {i} as vector {vector_id}")

print("🎉 Done! All chunks embedded and stored in Pinecone.")
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# --- Explicitly Mention Pinecone Configuration ---
print("--- Pinecone Configuration ---")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medium-blogs-embedding-index")
print(f"Pinecone API Key (from .env): {'SET' if PINECONE_API_KEY else 'NOT SET'}")
print(f"Pinecone Environment (from .env): {'SET' if PINECONE_ENVIRONMENT else 'NOT SET'}")
print(f"Pinecone Index Name (from .env): {PINECONE_INDEX_NAME}")

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    print("Error: Pinecone API key and environment not found in .env file.")
    exit()

# --- Embedding Model Configuration ---
print("\n--- Embedding Model Configuration ---")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
print(f"Embedding Model: {EMBEDDING_MODEL}")
print(f"Embedding Dimensions: {EMBEDDING_DIMENSIONS}")

# --- Test Query ---
print("\n--- Test Query ---")
TEST_QUERY = "what is Pinecone in machine learning?"
print(f"Test Query: {TEST_QUERY}")

def test_pinecone_retrieval(query: str):
    """Tests the retrieval of relevant documents from Pinecone."""
    print("\n--- Starting Pinecone Retrieval Test ---")
    try:
        # Initialize Pinecone connection directly using pinecone client
        print("Initializing Pinecone client...")
        pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        print(f"Connected to Pinecone environment: {pinecone.describe_index(PINECONE_INDEX_NAME).environment if PINECONE_INDEX_NAME in pinecone.list_indexes().names else 'NOT FOUND'}")
        index = pinecone.Index(PINECONE_INDEX_NAME)
        print(f"Using Pinecone index: {index.name}")

        # Initialize OpenAI Embeddings (used for querying)
        print("\nInitializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS
        )
        print("OpenAI Embeddings initialized.")

        # Create the Pinecone Vector Store
        print("\nCreating Pinecone Vector Store...")
        vectorstore = PineconeVectorStore(
            index,
            embeddings,
            text_key="chunk"  # Assuming your text is in the 'chunk' field
        )
        print("Pinecone Vector Store created.")

        # Perform the retrieval
        print("\nPerforming retrieval from Pinecone...")
        retriever = vectorstore.as_retriever()
        retrieved_documents = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(retrieved_documents)} documents from Pinecone.")

        print(f"\n--- Retrieval Test Results for Query: '{query}' ---")
        if retrieved_documents:
            for i, doc in enumerate(retrieved_documents):
                print(f"\nDocument {i+1} (from Pinecone):")
                print(f"  Metadata: {doc.metadata}")
                print(f"  Content (first 100 chars): '{doc.page_content[:100]}...'")
        else:
            print("No documents were retrieved from Pinecone for this query.")
        print("--- End of Retrieval Test ---")

    except Exception as e:
        print(f"An error occurred during Pinecone retrieval: {e}")
    finally:
        if 'pinecone' in locals():
            print("\nClosing Pinecone connection.")
            pinecone.close()

if __name__ == "__main__":
    test_pinecone_retrieval(TEST_QUERY)
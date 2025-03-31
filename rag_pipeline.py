import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def run_rag_pipeline(query):
    print("📥 Received query:", query)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    llm = ChatOpenAI(
        temperature=0,
        model_name="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.environ.get("DEEPSEEK_API_KEY")
    )

    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )

    template = """Use the following pieces of context to answer the question at the end.

If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "That is an interesting question!" at the beginning of the answer.

{context}

Question: {question}

Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke(query)
    print("✅ RAG result:", result)

    try:
        return result.content
    except AttributeError:
        return str(result)

# rag.py
# Loads the vector database, retrieves the most relevant chunks for a
# question, and asks Llama 3 to answer using ONLY those chunks.

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. Reconnect to the Chroma DB we built in ingest.py (same embedding model).
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# 2. A retriever: given a question, fetch the top-k most similar chunks.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. The chat model that writes the answer (your local Llama 3).
llm = ChatOllama(model="llama3", temperature=0)

# 4. The prompt: we INSTRUCT the model to answer only from the context.
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the
context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

def ask(question: str):
    # Retrieve relevant chunks.
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Fill the prompt and send it to Llama 3.
    message = prompt.format(context=context, question=question)
    response = llm.invoke(message)

    print("\n--- RETRIEVED CONTEXT ---")
    print(context)
    print("\n--- ANSWER ---")
    print(response.content)

# Try it.
if __name__ == "__main__":
     ask("How much is the home office stipend?")
# ingest.py
# Reads a document, splits it into chunks, turns each chunk into an
# embedding (numbers that capture meaning), and stores them in a local
# Chroma database so we can later search by meaning.

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. LOAD — read the raw text from our document.
loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

# 2. SPLIT — break it into smaller overlapping chunks.
#    chunk_size: max characters per chunk.
#    chunk_overlap: characters shared between neighbours so we don't
#    slice a sentence's meaning in half at the boundary.
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# 3. EMBED — the model that converts text to vectors (runs locally via Ollama).
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. STORE — embed every chunk and save into a local Chroma DB on disk.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)

print("Done. Chunks embedded and stored in chroma_db/")
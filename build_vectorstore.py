import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KNOWLEDGE_DIR = "knowledge_base"
PERSIST_DIR = "chroma_db"

print("Loading documents...")
loader = DirectoryLoader(KNOWLEDGE_DIR, glob="*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunk(s)")

print("Building embeddings (first run downloads a small model, ~90MB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)

print(f"Vector store built and saved to '{PERSIST_DIR}'")
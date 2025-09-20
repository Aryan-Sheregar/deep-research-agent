from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_model_embeddings(model="gemma:2b"):
    return OllamaEmbeddings(model=model)


def get_text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def get_vector_store(embeddings, persist_directory="chroma_db"):
    return Chroma(embedding_function=embeddings, persist_directory=persist_directory)


def add_context_to_vector_store(vector_store, text_splitter, source_text: str, source_url: str):
    if not source_text:
        print(f"No text to add for {source_url}")
        return
    else:
        chunks = text_splitter.split_text(source_text)
        vector_store.add_texts(
            chunks, metadatas=[{"source": source_url}] * len(chunks))
        print(
            f"Added {len(chunks)} chunks from {source_url} to the vector store.")

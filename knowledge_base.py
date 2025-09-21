from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

"""
Cached resources for embeddings, text splitter, and vector store. As when you start a new research the old vector store is deleted, these caches need to be cleared as well. 
"""

@st.cache_resource(show_spinner=False)
def get_model_embeddings(model="gemma:2b"):
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def get_text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@st.cache_resource(show_spinner=False)
def get_vector_store(persist_directory="chroma_db"):
    embeddings = get_model_embeddings()
    client = chromadb.PersistentClient(path=persist_directory)

    vector_store = Chroma(
        client=client,
        collection_name="research_collection",
        embedding_function=embeddings,
    )
    return vector_store


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


def safe_clear_vector_store(persist_directory="chroma_db", collection_name="research_collection"):

    try:
        client = chromadb.PersistentClient(path=persist_directory)

        collections = client.list_collections()
        if any(c.name == collection_name for c in collections):
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(name=collection_name)

        get_vector_store.clear()
        print("Vector store cache cleared.")

    except Exception as e:
        print(f"Error while trying to clear the vector store: {e}")

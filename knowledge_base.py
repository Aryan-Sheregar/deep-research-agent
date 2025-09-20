from langchain_community.embeddings import GemmaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_model_embeddings(model_name="gemma:2b"):
    return GemmaEmbeddings(model_name=model_name)

def get_text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def get_vector_store(embeddings, persist_directory="chroma_db"):
    return Chroma(embedding_function=embeddings, persist_directory=persist_directory)

#Testing
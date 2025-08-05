from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def split_documents(docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

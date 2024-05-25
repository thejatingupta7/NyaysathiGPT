# Import necessary modules from LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path to PDF documents
DATASET = "data_law/"

# Path to save the FAISS index
FAISS_INDEX = "embed_db/"                # folder containing embeddings (vectorised information)

# Function to embed all files in the dataset directory
def embed_all():

    # Initialize the document loader to load PDFs from the directory
    loader = DirectoryLoader(DATASET, glob="*.pdf", loader_cls=PyPDFLoader)
    
    # Load the PDF documents
    documents = loader.load()
    
    # Initialize the text splitter to create chunks of the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    
    # Split the documents into chunks
    chunks = splitter.split_documents(documents)
    
    # Initialize the embeddings generator
    embeddings = HuggingFaceEmbeddings()
    
    # Create the vector store using FAISS with the document chunks and their embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store locally
    vector_store.save_local(FAISS_INDEX)

# Execute the embedding function if the script is run directly
if __name__ == "__main__":
    embed_all()
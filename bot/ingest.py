from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_document():
    try:
        loader = DirectoryLoader("knowledge_base", glob="*.md")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40
        )
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="chroma_db",
            collection_name="hellobooks"
        )

        print("Documents embedded successfully!")

    except Exception as e:
        print("Error in ingest_document:", e)

if __name__ == "__main__":
    ingest_document()
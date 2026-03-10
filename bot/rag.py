from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_retrival():
    try:
        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load vector database
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
            collection_name="hellobooks"
        )

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        return retriever

    except Exception as e:
        print("❌ Error loading retrieval system:")
        print(e)
        return None


def retrieve_documents(query):
    try:
        retriever = load_retrival()

        if retriever is None:
            print("❌ Retriever could not be initialized.")
            return []

        docs = retriever.invoke(query)

        return docs

    except Exception as e:
        print("❌ Error retrieving documents:")
        print(e)
        return []
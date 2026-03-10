

import streamlit as st
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

st.title("📊 Hellobooks AI Assistant")
st.write("Ask accounting questions related to bookkeeping, invoices, profit & loss, etc.")

try:
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector database
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

except Exception as e:
    st.error("Error loading vector database or embeddings.")
    st.exception(e)

try:
    # Load LLM
    llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    st.error("Error loading the language model. Check your API key.")
    st.exception(e)

query = st.text_input("Ask a question:")

if query:
    try:
        with st.spinner("Thinking..."):

            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are an accounting assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""


            response = llm.invoke(prompt)

            answer = response.content

            # Remove reasoning if present
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()

            st.subheader("Answer")
            st.write(answer)

    except Exception as e:
        st.error("Something went wrong while generating the answer.")
        st.exception(e)
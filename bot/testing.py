from rag import retrieve_documents

docs = retrieve_documents("What is bookkeeping?")

for doc in docs:
    print(doc.page_content)
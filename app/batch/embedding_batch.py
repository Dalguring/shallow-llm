import os
import chromadb
import app.config
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.loader.file_loader import FileLoader


def run_embedding_batch(directory):
    client = chromadb.PersistentClient(os.getenv("VECTORDB_PATH"))
    collection = client.get_or_create_collection(os.getenv("COLLECTION"))

    for filename in os.listdir(directory):
        print(f"Processing file: {filename}")
        file_path = os.path.join(directory, filename)
        loader = FileLoader(file_path)
        results = collection.get(where={"source": file_path})

        if len(results['ids']) != 0:
            try:
                collection.delete(ids=results['ids'])
                print("Document deleted successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")

        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True})

        if filename.endswith(".pdf"):
            docs = loader.pdf_loader()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=30)
            split_texts = text_splitter.split_documents(docs)
            Chroma.from_documents(documents=split_texts, embedding=embeddings, collection_name=os.getenv("COLLECTION"),
                                  client=client)
        elif filename.endswith(".xlsx"):
            docs = loader.excel_to_doc()
            Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=os.getenv("COLLECTION"),
                                  client=client)

        elif filename.endswith(".csv"):
            docs = loader.csv_loader()
            Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=os.getenv("COLLECTION"),
                                  client=client)

        os.remove(file_path)


if __name__ == "__main__":
    run_embedding_batch(directory=os.getenv("RESOURCE_FILE_PATH"))

import os
import chromadb
import config
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

        os.remove(file_path)


if __name__ == "__main__":
    run_embedding_batch(directory=os.getenv("RESOURCE_FILE_PATH"))

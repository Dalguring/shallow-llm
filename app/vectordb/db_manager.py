import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import app.config


class VectorDBManager:
    def __init__(self, persist_client=os.getenv("VECTORDB_PATH")):
        self.persist_client = persist_client

    def get_vectorstore(self):
        client = chromadb.PersistentClient(self.persist_client)

        vectorstore = Chroma(client=client, collection_name=os.getenv("COLLECTION"),
                             embedding_function=HuggingFaceEmbeddings(
                                 model_name=os.getenv("EMBEDDING_MODEL_NAME"),
                                 model_kwargs={'device': 'cpu'},
                                 encode_kwargs={'normalize_embeddings': True}))

        return vectorstore

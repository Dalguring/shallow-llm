import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


class FileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def pdf_loader(self):
        loader = PyPDFLoader(self.file_path)
        data = loader.load()
        return data

    def csv_loader(self):
        loader = CSVLoader(self.file_path, encoding='cp949')
        data = loader.load()
        return data

    def text_loader(self):
        loader = TextLoader(self.file_path)
        data = loader.load()
        return data

    def excel_to_doc(self):
        df = pd.read_excel(self.file_path)
        documents = []
        for index, row in df.iterrows():
            content = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            doc = Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "row": index,
                }
            )
            documents.append(doc)
        return documents


from transformers import pipeline
import torch
from app.models.tokenizer_model import TokenizerModel
from app.vectordb.db_manager import VectorDBManager
from app.prompt.prompt_template import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from app.batch.embedding_batch import run_embedding_batch
import app.config
import time
import os


class LLMModel:
    def __init__(self):
        run_embedding_batch(directory=os.getenv("RESOURCE_FILE_PATH"))

        self.tokenizer, self.model = TokenizerModel().load_model()
        self.vectorstore = VectorDBManager().get_vectorstore()
        self.prompt = PromptTemplate().get_prompt()
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def generate_response(self, user_question: str) -> str:
        retriever = self.vectorstore.as_retriever()

        def format_docs(docs):
            return '\n\n'.join(doc.page_content for doc in docs)

        rag_chain = (
                {'context': retriever | format_docs, 'question': RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

        with torch.no_grad():
            start_time = time.time()
            result = rag_chain.invoke(user_question)
            time_taken = time.time() - start_time

        return result, time_taken

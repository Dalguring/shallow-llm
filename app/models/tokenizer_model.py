from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import app.config


class TokenizerModel:
    def load_model(self):
        tokenizer_path = os.getenv("MODEL_PATH") + os.getenv("MODEL_NAME")
        model_path = os.getenv("MODEL_PATH") + os.getenv("MODEL_NAME")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        return tokenizer, model

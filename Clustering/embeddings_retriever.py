from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers.utils import logging

class EmbeddingsRetriever:
    def __init__(self, model_name):
        logging.set_verbosity_error()
        # Checking if the model is a sentencen transformer
        if "sentence-transformers" in model_name:
            self.model = SentenceTransformer(model_name)
            self.model_type = "sentence-transformer"
        else:
            # Using huggingface pipeline for feature extraction for other models
            self.pipeline = pipeline('feature-extraction', model=model_name, truncation=True, padding=True)
            self.model_type = "huggingface"

    def get_embeddings(self, text: str) -> np.ndarray:
        if self.model_type == "sentence-transformer":
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Using huggingface pipeline for feature extraction
            embedding = self.pipeline(text)
            embedding = np.mean(embedding[0], axis=0)  # Averaging across tokens to get the sentence embedding

        return embedding

    # gets embeddings for multiple texts
    def get_multiple_embeddings(self, texts: list[str]) -> dict[str, np.ndarray]:
        embeddings_dict = {}
        for text in texts:
            embeddings_dict[text] = self.get_embeddings(text)
        return embeddings_dict
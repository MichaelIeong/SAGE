from functools import lru_cache
from langchain.embeddings.base import Embeddings
import requests


class OllamaEmbeddingOnly(Embeddings):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        response = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": text
        })
        response.raise_for_status()
        return response.json()["embedding"]
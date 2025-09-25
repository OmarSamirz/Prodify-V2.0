import torch
from torch import Tensor
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import os
from typing import List, Optional

from constants import DTYPE_MAP, EMBEDDING_MODEL_PATH

load_dotenv()


class SentenceEmbeddingModel:

    def __init__(self):
        super().__init__()
        self.model_id = os.getenv("E_MODEL_NAME")
        self.device = torch.device(os.getenv("E_DEVICE"))
        self.show_progrees_bar = eval(os.getenv("E_SHOW_PROGRESS_BAR"))
        self.convert_to_numpy = eval(os.getenv("E_CONVERT_TO_NUMPY"))
        self.convert_to_tensor = eval(os.getenv("E_CONVERT_TO_TENSOR"))
        dtype = os.getenv("E_DTYPE")
        if dtype in DTYPE_MAP:
            self.dtype = DTYPE_MAP[dtype]
        else:
            raise ValueError(f"This dtype {dtype} is not supported.")

        try:
            self.model = SentenceTransformer(
                str(EMBEDDING_MODEL_PATH),
                device=self.device,
            )
        except:
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device
            )

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name,
            convert_to_numpy=self.convert_to_numpy,
            convert_to_tensor=self.convert_to_tensor,
            show_progress_bar=self.show_progrees_bar,
        )
        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        return self.model.similarity(query_embeddings, document_embeddings)

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "query")
        document_embeddings = self.get_embeddings(documents)
        return self.calculate_scores(query_embeddings, document_embeddings)
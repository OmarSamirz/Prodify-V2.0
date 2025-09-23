import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import os
from typing import Optional, List, Union
from typing_extensions import override

from constants import MODEL_PATH
from modules.logger import logger
from models import TfidfBaseModel

load_dotenv()


class BrandsClassifier(TfidfBaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = os.getenv("TS_MODEL_NAME")
        try:
            self.df_brands = pd.read_csv(os.getenv("TS_BRANDS_CSV_PATH"))
            self.df_brands["documents"] = self.df_brands["Sector"] + " " + self.df_brands["Brand"] + " " + self.df_brands["Product"]
            self.documents = self.df_brands["documents"].tolist()
        except:
            self.documents = None
        if os.path.exists(MODEL_PATH / self.model_name):
            self.load()

    @override
    def fit(self, documents: Optional[List[str]] = None) -> None:
        if documents is not None:
            self.documents = documents

        self.vectorizer.fit(self.documents)

    def get_vector(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]

        return self.vectorizer.transform(text)

    def get_similarity(self, query_vector: np.ndarray, documents_vectors: np.ndarray) -> np.ndarray:
        return cosine_similarity(query_vector, documents_vectors)

    @override
    def predict(self, query: Union[str, List[str]]) -> List[str]:
        if isinstance(query, str):
            query = [query]

        query_vector = self.get_vector(query)
        documents_vectors = self.get_vector(self.documents)

        logger.info("Predicting `segments`, `families` and `classes` for invoice(s) in `Brands Classifer`.")
        similarities = self.get_similarity(query_vector, documents_vectors)
        indices = np.argmax(similarities, axis=1)
        products = self.df_brands.iloc[indices]
        seg = products["Segment"].tolist()
        fam = products["Family"].tolist()
        cls = products["Class"].tolist()

        return [seg, fam, cls]

    @override
    def save(self) -> None:
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH, exist_ok=True)

        model_path = MODEL_PATH / self.model_name
        joblib.dump(self.vectorizer, model_path)

    @override
    def load(self) -> None:
        if not os.path.exists(MODEL_PATH / self.model_name):
            raise ValueError("You need to fit the model first.")

        model_path = MODEL_PATH / self.model_name
        self.vectorizer = joblib.load(model_path)

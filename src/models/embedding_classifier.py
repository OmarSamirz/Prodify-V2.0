import torch
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

import os
from typing import List, Union, Optional, Tuple

from modules.logger import logger
from models.sentence_embedding_model import SentenceEmbeddingModel

load_dotenv()


class EmbeddingClassifier:

    def __init__(self, embedding_model: SentenceEmbeddingModel):
        self.embedding_model = embedding_model
        self.topk = int(os.getenv("EC_TOP_K"))
        self.df_gpc = pd.read_csv(os.getenv("GPC_CSV_PATH"))

    def classify(self, product_name: Union[str, List[str]], labels: List[str], is_max: bool = True) -> Union[str, List[str]]:
        if len(labels) == 1:
            return labels[0]

        scores = self.embedding_model.get_scores(product_name, labels)
        idx = torch.argmax(scores, dim=1) if is_max else torch.argmin(scores, dim=1)
        if isinstance(product_name, List):
            return [labels[i] for i in idx]

        return labels[idx]

    def get_gpc(
        self,
        products_name: Union[str, List[str]],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        pred_segments, pred_families, pred_classes = [], [], []
        if isinstance(products_name, str):
            products_name = [products_name]

        if labels is None:
            labels = self.df_gpc["SegmentTitle"].drop_duplicates().tolist()

        logger.info("Predicting `segments` for invoice(s) in `Embedding Classifer`.")
        pred_segments = self.classify(products_name, labels)
        logger.info("Predicting `families` for invoice(s) in `Embedding Classifer`.")
        for prod, seg in tqdm(zip(products_name, pred_segments), total=len(products_name)):
            fam_candidates = self.df_gpc[self.df_gpc["SegmentTitle"]==seg]["FamilyTitle"].drop_duplicates().tolist()
            pred_families.append(self.classify(prod, fam_candidates))

        logger.info("Predicting `classes` for invoice(s) in `Embedding Classifer`.")
        for prod, fam in tqdm(zip(products_name, pred_families), total=len(products_name)):
            cls_candidates = self.df_gpc[self.df_gpc["FamilyTitle"]==fam]["ClassTitle"].drop_duplicates().tolist()
            pred_classes.append(self.classify(prod, cls_candidates))

        return pred_segments, pred_families, pred_classes

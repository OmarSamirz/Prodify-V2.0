from dotenv import load_dotenv

import os
from collections import Counter
from typing import List, Tuple, Dict, Union, Any

from models.tfidf_classifier import TfidfClassifier
from models.brands_classifier import BrandsClassifier
from models.embedding_classifier import EmbeddingClassifier

load_dotenv()


class EnsembleModel:

    def __init__(
        self,
        brands_classifier: BrandsClassifier,
        embedding_classifier: EmbeddingClassifier,
        tfidf_classifier: TfidfClassifier
    ) -> None:
        self.brand_tfidf_similiraity = brands_classifier
        self.embed_clf = embedding_classifier
        self.tfidf_clf = tfidf_classifier
        self.num_models = int(os.getenv("EM_NUM_MODELS"))
        self.df_gpc = self.embed_clf.df_gpc

    def extract_labels(self, cls_label: str) -> Tuple[str, str]:
        classes = self.df_gpc["ClassTitle"].unique().tolist()
        if cls_label in classes:
            seg = self.df_gpc[self.df_gpc["ClassTitle"]==cls_label]["SegmentTitle"].tolist()[0]
            fam = self.df_gpc[self.df_gpc["ClassTitle"]==cls_label]["FamilyTitle"].tolist()[0]

        return seg, fam

    def predict(self, invoice_items: Union[str, List[str]]) -> Dict[str, Any]:
        brand_tfidf_similiraity_pred = self.brand_tfidf_similiraity.predict(invoice_items)
        tfidf_clf_pred = self.tfidf_clf.predict(invoice_items)
        embed_clf_pred = self.embed_clf.get_gpc(invoice_items)

        return {
            "embed_clf": embed_clf_pred,
            "brand_tfidf_sim": brand_tfidf_similiraity_pred,
            "tfidf_clf": tfidf_clf_pred
        }

    def vote(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        pred_classes = []
        results = {
            "voted_segments": [],
            "voted_families": [],
            "voted_classes": [],
            "confidences": [],
        }
        pred_classes.append(predictions["embed_clf"][2])
        pred_classes.append(predictions["brand_tfidf_sim"][2])
        pred_classes.append(predictions["tfidf_clf"][2])

        for i in range(len(pred_classes[0])):
            classes = []
            classes.append(pred_classes[0][i])
            classes.append(pred_classes[1][i])
            classes.append(pred_classes[2][i])

            cls_counter = Counter(classes)
            voted_cls, cls_count = cls_counter.most_common(1)[0]
            if cls_count < 2:
                voted_cls = pred_classes[2][i]

            voted_seg, voted_fam = self.extract_labels(voted_cls)
            results["voted_segments"].append(voted_seg)
            results["voted_families"].append(voted_fam)
            results["voted_classes"].append(voted_cls)
            results["confidences"].append(cls_count / self.num_models)

        results["embed_clf_preds"] = predictions["embed_clf"]
        results["brand_tfidf_sim_preds"] = predictions["brand_tfidf_sim"]
        results["tfidf_clf_preds"] = predictions["tfidf_clf"]

        return results

    def run_ensemble(self, invoice_items: Union[str, List[str]]) -> Dict[str, Any]:
        preds = self.predict(invoice_items)
        voted = self.vote(preds)

        return voted
import torch
import joblib
import numpy as np
import pandas as pd
from torch import Tensor
from tqdm.auto import tqdm
from langid import classify
from dotenv import load_dotenv
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

import os
import ast
from collections import Counter
from abc import ABC, abstractmethod 
from typing_extensions import override
from typing import List, Optional, Dict, Any, Union, Tuple

from constants import MODEL_PATH, DTYPE_MAP

load_dotenv()


class OpusTranslationModel:

    def __init__(self):
        self.model_name = os.getenv("T_MODEL_NAME")
        self.device = torch.device(os.getenv("T_DEVICE"))
        self.truncation = eval(os.getenv("T_TRUNCATION"))
        self.padding = eval(os.getenv("T_PADDING"))
        self.skip_special_tokens = eval(os.getenv("T_SKIP_SPECIAL_TOKENS"))
        dtype = os.getenv("T_DTYPE")
        if dtype in DTYPE_MAP:
            self.dtype = DTYPE_MAP[dtype]

        self.model = MarianMTModel.from_pretrained(
            self.model_name, 
            device_map=self.device, 
            torch_dtype=self.dtype
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, text: str) -> str:
        lang, _ = classify(text)
        if lang == "en":
            return text
        
        tokens = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        ).to(self.device)
        translated_tokens = self.model.generate(**tokens)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=self.skip_special_tokens)

        return translated_text


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

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
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


class EmbeddingClassifier:

    def __init__(self):
        self.embed_model = SentenceEmbeddingModel()
        self.topk = int(os.getenv("EC_TOP_K"))
        self.df_gpc = pd.read_csv(os.getenv("GPC_CSV_PATH"))

    def classify(self, product_name: Union[str, List[str]], labels: List[str], is_max: bool = True) -> Union[str, List[str]]:
        if len(labels) == 1:
            return labels[0]

        scores = self.embed_model.get_scores(product_name, labels)
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

        pred_segments = self.classify(products_name, labels)
        for prod, seg in tqdm(zip(products_name, pred_segments), total=len(products_name)):
            fam_candidates = self.df_gpc[self.df_gpc["SegmentTitle"]==seg]["FamilyTitle"].drop_duplicates().tolist()
            pred_families.append(self.classify(prod, fam_candidates))

        for prod, fam in tqdm(zip(products_name, pred_families), total=len(products_name)):
            cls_candidates = self.df_gpc[self.df_gpc["FamilyTitle"]==fam]["ClassTitle"].drop_duplicates().tolist()
            pred_classes.append(self.classify(prod, cls_candidates))

        return pred_segments, pred_families, pred_classes


class TfidfBaseModel(ABC):

    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            analyzer=os.getenv("TB_ANALYZER"),
            ngram_range=ast.literal_eval(os.getenv("TB_NGRAM_RANGE")),
            min_df=int(os.getenv("TB_MIN_DF")),
            max_df=float(os.getenv("TB_MAX_DF")),
            lowercase=eval(os.getenv("TB_LOWERCASE")),
            sublinear_tf=eval(os.getenv("TB_SUBLINEAR_TF")),
            smooth_idf=eval(os.getenv("TB_SMOOTH_IDF")),
            norm=os.getenv("TB_NORM"),
            strip_accents=os.getenv("TC_STRIP_ACCENTS", None),
            stop_words=os.getenv("TC_STOP_WORDS", None),
            token_pattern=None if os.getenv("TB_ANALYZER") in ("char", "char_wb") else r'(?u)\b\w+\b',
        )

    @abstractmethod
    def fit(self, X_train, y_train) -> None:
        ...

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...
    

class TfidfClassifier(TfidfBaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = os.getenv("TC_MODEL_NAME")
        df_gpc = pd.read_csv(os.getenv("GPC_CSV_PATH"))
        self.df_gpc = df_gpc.groupby('ClassTitle')[['SegmentTitle', 'FamilyTitle']].first().reset_index()
        self.svm = LinearSVC(
            C=float(os.getenv("TC_C")),
            class_weight=os.getenv("TC_CLASS_WEIGHT")
        )

        self.clf = None

    @override
    def fit(self, X_train, y_train) -> None:
        self.clf = Pipeline(
            [
                ("vectorizer_tfidf", self.vectorizer),
                ("svm", self.svm)
            ]
        )
        self.clf.fit(X_train, y_train)

    @override
    def predict(self, x: Union[str, List[str]]) -> List[str]:
        if isinstance(x, str):
            x = [x]

        cls = self.clf.predict(x).tolist()
        df_preds = pd.DataFrame({"ClassTitle": cls})
        df_merged = pd.merge(df_preds, self.df_gpc, on="ClassTitle", how="left")
        seg = df_merged["SegmentTitle"].tolist()
        fam = df_merged["FamilyTitle"].tolist()


        return [seg, fam, cls]

    @override
    def save(self) -> None:
        if self.clf is None:
            raise ValueError("You need to fit the model first.")
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH, exist_ok=True)

        model_path = MODEL_PATH / self.model_name
        joblib.dump(self.clf, model_path)

    @override
    def load(self) -> None:
        if self.clf is not None:
            return

        model_path = MODEL_PATH / self.model_name
        self.clf = joblib.load(model_path)


class TfidfSimilarityModel(TfidfBaseModel):

    def __init__(self):
        super().__init__()
        self.model_name = os.getenv("TS_MODEL_NAME")
        try:
            self.df_brands = pd.read_csv(os.getenv("TS_BRANDS_CSV_PATH"))
            self.df_brands["documents"] = self.df_brands["Sector"] + " " + self.df_brands["Brand"] + " " + self.df_brands["Product"]
            self.documents = self.df_brands["documents"].tolist()
        except:
            self.documents = None

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


class EnsembleModel:

    def __init__(self):
        self.brand_tfidf_similiraity = TfidfSimilarityModel()
        self.embed_clf = EmbeddingClassifier()
        self.tfidf_clf = TfidfClassifier()
        self.tfidf_clf.load()
        self.brand_tfidf_similiraity.load()
        self.num_models = int(os.getenv("EM_NUM_MODELS"))
        self.df_gpc = self.embed_clf.df_gpc

    def extract_labels(self, cls_label: str) -> Tuple[str, str]:
        classes = self.df_gpc["ClassTitle"].unique().tolist()
        if cls_label in classes:
            seg = self.df_gpc[self.df_gpc["ClassTitle"]==cls_label]["SegmentTitle"].tolist()[0]
            fam = self.df_gpc[self.df_gpc["ClassTitle"]==cls_label]["FamilyTitle"].tolist()[0]

        return seg, fam

    def predict(self, product_name: str) -> Dict[str, Any]:
        brand_tfidf_similiraity_pred = self.brand_tfidf_similiraity.predict(product_name)
        tfidf_clf_pred = self.tfidf_clf.predict(product_name)
        embed_clf_pred = self.embed_clf.get_gpc(product_name)

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
            results["voted_segments"].extend(voted_seg)
            results["voted_families"].extend(voted_fam)
            results["voted_classes"].extend(voted_cls)
            results["confidences"].extend(cls_count / self.num_models)

        results["embed_clf_preds"] = predictions["embed_clf"]
        results["brand_tfidf_sim_preds"] = predictions["brand_tfidf_sim"]
        results["tfidf_clf_preds"] = predictions["tfidf_clf"]

        return results

    def run_ensemble(self, invoice_item: str) -> Dict[str, Any]:
        preds = self.predict(invoice_item)
        voted = self.vote(preds)

        return voted
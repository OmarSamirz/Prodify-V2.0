import torch
import joblib
import pandas as pd
import xgboost as xgb
from torch import Tensor
from langid import classify
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
from transformers import BitsAndBytesConfig

import os
import json
from dataclasses import dataclass
from typing_extensions import override
from typing import List, Optional, Dict, Any, Union

from constants import MODEL_PATH, RANDOM_STATE, DTYPE_MAP


@dataclass
class OpusTranslationModelConfig:
    padding: bool
    model_name: str
    device: str
    dtype: str
    truncation: bool
    skip_special_tokens: bool


class OpusTranslationModel:

    def __init__(self, config: OpusTranslationModelConfig):
        self.config = config
        self.model = MarianMTModel.from_pretrained(
            self.config.model_name, 
            device_map=self.config.device, 
            torch_dtype=self.config.dtype
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
        
    def translate(self, text: str) -> str:
        lang, _ = classify(text)
        if lang == "en":
            return text
        
        tokens = self.tokenizer(
            text,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt"
        ).to(self.config.device)
        translated_tokens = self.model.generate(**tokens)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=self.config.skip_special_tokens)

        return translated_text


@dataclass
class SentenceEmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    truncate_dim: Optional[int]
    convert_to_numpy: bool
    convert_to_tensor: bool
    use_prompt: bool = False
    prompt_config: Optional[Dict[str, str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None


class SentenceEmbeddingModel:
    
    def __init__(self, config: SentenceEmbeddingConfig):
        super().__init__()
        self.config = config
        self.model_id = config.model_id
        self.device = config.device
        self.truncate_dim = config.truncate_dim
        if config.dtype in DTYPE_MAP:
            self.dtype = DTYPE_MAP[config.dtype]
        else:
            raise ValueError(f"This dtype {config.dtype} is not supported.")

        model_kwargs = config.model_kwargs or {}

        if "quantization_config" in model_kwargs:
            quant_config = model_kwargs["quantization_config"]
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant_config.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True)
            )

        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            model_kwargs=model_kwargs
        )

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        if self.config.use_prompt and prompt_name and self.config.prompt_config:
            if prompt_name in self.config.prompt_config:
                prompt_template = self.config.prompt_config[prompt_name]
                texts = [prompt_template.format(text=t) for t in texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=self.config.convert_to_numpy,
            convert_to_tensor=self.config.convert_to_tensor,
            show_progress_bar=True
        )
        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        return self.model.similarity(query_embeddings, document_embeddings)

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "query")
        document_embeddings = self.get_embeddings(documents)
        return self.calculate_scores(query_embeddings, document_embeddings)


@dataclass
class EmbeddingClassifierConfig:
    embed_model_config: SentenceEmbeddingConfig
    topk: int
    gpc_csv_path: str


class EmbeddingClassifier:

    def __init__(self, config: EmbeddingClassifierConfig):
        self.embed_model = SentenceEmbeddingModel(config.embed_model_config)
        self.topk = config.topk
        self.df_gpc = pd.read_csv(config.gpc_csv_path)

    def classify(self, product_name: Union[str, List[str]], labels: List[str], is_max: bool = True) -> Union[str, List[str]]:
        if len(labels) == 1:
            return labels[0]

        scores = self.embed_model.get_scores(product_name, labels)
        idx = torch.argmax(scores, dim=1) if is_max else torch.argmin(scores, dim=1)
        if isinstance(product_name, List):
            return [labels[i] for i in idx]

        return labels[idx]

    def classify_topk(self, product_name: Union[str, List[str]], labels: List[str]) -> Union[List[str], List[List[str]]]:
        if len(labels) == 1:
            return labels

        scores = self.embed_model.get_scores(product_name, labels)
        k = min(self.topk, scores.size(1))
        _, topk_indices = torch.topk(scores, dim=1, k=k)
        topk_indices = topk_indices.squeeze(0)
        if isinstance(product_name, List):
            return [labels[i][j] for i in topk_indices[0] for j in topk_indices[1]]

        topk_labels = [labels[i] for i in topk_indices]

        return topk_labels
    
    def get_gpc(
        self, 
        product_name: str, 
        labels: Optional[List[str]] = None, 
        level: str = "segment", 
        is_topk: bool = True
    ) -> List[str]:
        pred_labels = []
        if level == "segment":
            if labels is None:
                labels = self.df_gpc["SegmentTitle"].drop_duplicates().tolist()

            seg_label = self.classify(product_name, labels)
            candidates = self.df_gpc[self.df_gpc["SegmentTitle"] == seg_label]["FamilyTitle"].drop_duplicates().tolist()
            pred_labels.append(seg_label)
            pred_labels.extend(self.get_gpc(product_name, candidates, "family"))

        elif level == "family":
            fam_label = self.classify(product_name, labels)
            candidates = self.df_gpc[self.df_gpc["FamilyTitle"] == fam_label]["ClassTitle"].drop_duplicates().tolist()
            pred_labels.append(fam_label)
            pred_labels.extend(self.get_gpc(product_name, candidates, "class"))

        elif level == "class":
            cls_label = self.classify(product_name, labels)
            candidates = self.df_gpc[self.df_gpc["ClassTitle"] == cls_label]["BrickTitle"].drop_duplicates().tolist()
            pred_labels.append(cls_label)
            pred_labels.extend(self.get_gpc(product_name, candidates, "brick"))

        elif level == "brick":
            brk_label = self.classify_topk(product_name, labels) if is_topk else self.classify(product_name, labels)
            pred_labels.extend(brk_label) if is_topk else pred_labels.append(brk_label)

        else:
            raise ValueError(f"Level `{level}` is not supported.")

        return pred_labels
    
    def predict_brick_by_exclusion(
        self, 
        product_name: str, 
        candidate_bricks: List[str],
        exclusion_column: str = "BrickDefinition_Excludes"
    ) -> str:
        
        candidate_df = self.df_gpc[
            self.df_gpc["BrickTitle"].isin(candidate_bricks)
        ].copy()
       
        null_exclusions = candidate_df[
            candidate_df[exclusion_column].isnull() | 
            (candidate_df[exclusion_column].astype(str).str.strip() == "")
        ]

        if not null_exclusions.empty:
            return null_exclusions.iloc[0]["BrickTitle"]
        
        exclusion_texts = candidate_df[exclusion_column].tolist()

        least_excluison = self.classify(product_name, exclusion_texts, False)
        top_brick = candidate_df[candidate_df[exclusion_column]==least_excluison]["BrickTitle"].values.item()

        # inclusion_texts = candidate_df["BrickDefinition_Includes"].tolist()
        # concatenated_texts = [e + " " + i for e, i in zip(exclusion_texts, inclusion_texts)]
        # least_excluison = self.classify(product_name, concatenated_texts)

        # candidate_df["concat_text"] = (
        #     candidate_df[exclusion_column] + " " + candidate_df["BrickDefinition_Includes"]
        # )

        # top_brick = candidate_df.loc[candidate_df["concat_text"] == least_excluison, "BrickTitle"].iloc[0]
        
        return top_brick


@dataclass
class BrandEmbeddingClassifierConfig:
    embed_classifer_config: EmbeddingClassifierConfig
    brand_json_path: str


class BrandEmbeddingClassifier(EmbeddingClassifier):

    def __init__(self, config: BrandEmbeddingClassifierConfig):
        super().__init__(config.embed_classifer_config)
        with open(config.brand_json_path, "r") as f:
            self.brand_dataset = json.load(f)

        self.brand_names = list(self.brand_dataset.keys())
        brand_names_lower = [name.lower() for name in self.brand_names]

        self.token_to_brand = {}
        for idx, brand_name in enumerate(brand_names_lower):
            for token in brand_name.split():
                self.token_to_brand.setdefault(token, []).append(self.brand_names[idx])

    def get_brand_data(self, product_name: str) -> List[Dict[str, Any]]:
        tokens = product_name.lower().split()

        for token in tokens:
            if token in self.token_to_brand:
                brand = self.token_to_brand[token][0]
                print(f"The brand is {brand}")
                return self.brand_dataset[brand]

        return []
    
    @override
    def get_gpc(
        self, 
        product_name: str, 
        labels: Optional[List[str]] = None, 
        level: str = "segment", 
        is_topk: bool = False
    ) -> List[str]:
        pred_labels = []
        brand_data = self.get_brand_data(product_name)
        if not brand_data:
            return []

        if level == "segment":
            segments = [entry["Segment"] for entry in brand_data]
            seg_label = self.classify(product_name, segments)
            pred_labels.append(seg_label)
            families = [entry["Family"] for entry in brand_data if entry["Segment"] == seg_label]
            if families:
                pred_labels.extend(self.get_gpc(product_name, families, level="family", is_topk=is_topk))

        elif level == "family":
            fam_label = self.classify(product_name, labels)
            pred_labels.append(fam_label)
            classes = []
            for entry in brand_data:
                if entry["Family"] == fam_label:
                    classes.extend(entry["Class"])
            if classes:
                pred_labels.extend(self.get_gpc(product_name, classes, level="class", is_topk=is_topk))

        elif level == "class":
            cls_label = self.classify(product_name, labels)
            pred_labels.append(cls_label)
            bricks = []
            for entry in brand_data:
                if cls_label in entry["Class"]:
                    bricks.extend(entry["Brick"])
            if bricks:
                pred_labels.extend(self.get_gpc(product_name, bricks, level="brick", is_topk=is_topk))

        elif level == "brick":
            if not labels:
                return pred_labels
            if is_topk:
                brk_labels = self.classify_topk(product_name, labels)
                pred_labels.extend(brk_labels)
            else:
                brk_label = self.classify(product_name, labels)
                pred_labels.append(brk_label)

        else:
            raise ValueError(f"Level `{level}` is not supported.")

        return pred_labels


@dataclass
class TfidfClassifierConfig:
    model_name: str
    analyzer: str
    ngram_range: tuple
    min_df: int
    max_df: float
    lowercase: bool
    sublinear_tf: bool
    smooth_idf: bool
    norm: str
    strip_accents: str
    stop_words: set
    C: float
    class_weight: str


class TfidfClassifier:

    def __init__(self, config: Optional[TfidfClassifierConfig]):
        self.model_name = config.model_name
        self.vectorizer = TfidfVectorizer(
            analyzer=config.analyzer,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            lowercase=config.lowercase,
            sublinear_tf=config.sublinear_tf,
            smooth_idf=config.smooth_idf,
            norm=config.norm,
            strip_accents=config.strip_accents,
            stop_words=config.stop_words,
            token_pattern=None if config.analyzer in ("char", "char_wb") else r'(?u)\b\w+\b',
        )
        self.svm = LinearSVC(
            C=config.C,
            class_weight=config.class_weight
        )

        self.clf = None

    def fit(self, X_train, y_train) -> None:
        self.clf = Pipeline(
            [
                ("vectorizer_tfidf", self.vectorizer),
                ("svm", self.svm)
            ]
        )
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def save(self) -> None:
        if self.clf is None:
            raise ValueError("You need to fit the model first.")
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH, exist_ok=True)

        model_path = MODEL_PATH / self.model_name
        joblib.dump(self.clf, model_path)

    def load(self) -> None:
        if self.clf is not None:
            return

        model_path = MODEL_PATH / self.model_name
        self.clf = joblib.load(model_path)


@dataclass
class EmbeddingXGBoostConfig:
    embedding_config: SentenceEmbeddingConfig
    model_name: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    gamma: float
    reg_lambda: float
    reg_alpha: float
    objective: str
    n_jobs: int
    eval_metric: str


class EmbeddingXGBoostModel:

    def __init__(self, config: EmbeddingXGBoostConfig) -> None:
        self.model_name = config.model_name
        self.embedding_model = SentenceEmbeddingModel(config.embedding_config)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            gamma=config.gamma,
            reg_lambda=config.reg_lambda,
            reg_alpha=config.reg_alpha,
            objective=config.objective,
            n_jobs=config.n_jobs,
            eval_metric=config.eval_metric,
            random_state=RANDOM_STATE,
        )

    def fit(self, X_train, y_train):
        embeddings = self.embedding_model.get_embeddings(X_train)
        # self.clf = MultiOutputClassifier(self.xgb_model)
        self.xgb_model.fit(embeddings, y_train)

    def predict(self, x):
        embeddings = self.embedding_model.get_embeddings(x)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return self.xgb_model.predict(embeddings)
    
    def save(self) -> None:
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH, exist_ok=True)

        model_path = MODEL_PATH / self.model_name
        joblib.dump(self.xgb_model, model_path)

    def load(self) -> None:
        model_path = MODEL_PATH / self.model_name
        self.xgb_model = joblib.load(model_path)
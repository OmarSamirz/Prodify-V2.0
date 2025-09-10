import torch
import joblib
import xgboost as xgb
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from langid import classify
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
from transformers import BitsAndBytesConfig

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from constants import MODEL_PATH, RANDOM_STATE, DTYPE_MAP


@dataclass
class GpcHierarchicalClassifierConfig:
    model_name: str
    dtype: str
    device: str
    embedding_size: int
    hidden_size: int
    dropout: float
    segment_num_classes: int
    family_num_classes: int
    class_num_classes: int
    brick_num_classes: int


class GpcHierarchicalClassifier(nn.Module):

    def __init__(self, config: GpcHierarchicalClassifierConfig):
        super().__init__()
        self.model_name = config.model_name
        self.device = config.device

        if config.dtype in DTYPE_MAP:
            self.dtype = DTYPE_MAP[config.dtype]
        else:
            raise ValueError(f"This dtype {config.dtype} is not supported.")

        self.shared_head = nn.Sequential(
            nn.Linear(config.embedding_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.embedding_size)
        )

        self.segment_head = nn.Linear(config.embedding_size, config.segment_num_classes, bias=False)

        self.family_input_dim = config.embedding_size + config.segment_num_classes
        self.family_head = nn.Linear(self.family_input_dim, config.family_num_classes, bias=False)

        self.class_input_dim = config.embedding_size + config.segment_num_classes + config.family_num_classes
        self.class_head = nn.Linear(self.class_input_dim, config.class_num_classes, bias=False)

        self.brick_input_dim = config.embedding_size + config.segment_num_classes + config.family_num_classes + config.class_num_classes
        self.brick_head = nn.Linear(self.brick_input_dim, config.brick_num_classes, bias=False)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        shared = self.shared_head(x)

        segment_out = self.segment_head(x)

        family_inp = torch.cat([shared, segment_out], dim=1)
        family_out = self.family_head(family_inp)

        class_inp = torch.cat([shared, segment_out, family_out], dim=1)
        class_out = self.class_head(class_inp)

        brick_inp = torch.cat([shared, segment_out, family_out, class_out], dim=1)
        brick_out = self.brick_head(brick_inp)


        return {
            "segment": segment_out,
            "family": family_out,
            "class": class_out,
            "brick": brick_out
        }


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
        self.dtype = config.dtype
        self.truncate_dim = config.truncate_dim

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

        self.clf = None

    def fit(self, X_train, y_train) -> None:
        
        self.clf = Pipeline(
            [
                ("vectorizer_tfidf", self.vectorizer),
                ("xgboost", self.xgb_model)
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


@dataclass
class DummyModelConfig:
    strategy: str


class DummyModel:

    def __init__(self, config: DummyModelConfig):
        self.model = DummyClassifier(strategy=config.strategy)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import unicodedata
from sklearn.metrics import f1_score
from safetensors.torch import save_file
import pandas as pd
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
RANDOM_SEED = 42
from modules.models import EmbeddingClassifier, EmbeddingClassifierConfig

import re
import json
from typing import List, Union

from constants import MODEL_PATH
from modules.logger import logger

from modules.models import (
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
    OpusTranslationModel,
    OpusTranslationModelConfig,
    TfidfClassifier, 
    TfidfClassifierConfig,
    EmbeddingXGBoostConfig,
    EmbeddingXGBoostModel,
    GpcHierarchicalClassifierConfig,
    GpcHierarchicalClassifier,
    EmbeddingClassifier,
    EmbeddingClassifierConfig,
    BrandEmbeddingClassifier,
    BrandEmbeddingClassifierConfig,
    
)

def evaluation_score(y_true: List[str], y_pred: List[str], average: str) -> float:
    return f1_score(y_true, y_pred, average=average)

def remove_repeated_words(text):
    text = text.split()
    final_text = []
    for word in text:
        if word in final_text:
            continue
        final_text.append(word)

    return " ".join(final_text)

def remove_numbers(text: str, remove_string: bool = False) -> str:
    text = text.split()
    text = [t for t in text if not re.search(r"\d", t)] if remove_string else [re.sub(r"\d+", "", t) for t in text]

    return " ".join(text)

def remove_punctuations(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.strip().split())

def remove_extra_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text) -> str:
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_extra_space(text)

    return text.lower()

def load_embedding_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = SentenceEmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = SentenceEmbeddingModel(config)

    return model

def load_translation_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        config = OpusTranslationModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = OpusTranslationModel(config)

    return model

def load_embedding_classifier_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        embedding_config = SentenceEmbeddingConfig(**config_dict["embedding_config"])
        config = EmbeddingClassifierConfig(embedding_config, **config_dict["classification_config"])
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = EmbeddingClassifier(config)

    return model

def load_gpc_hierarchical_classifier(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        config = GpcHierarchicalClassifierConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = GpcHierarchicalClassifier(config)

    return model

def load_brand_embedding_classifier_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        embedding_config = SentenceEmbeddingConfig(**config_dict["embedding_config"])
        embedding_clf_config = EmbeddingClassifierConfig(embedding_config, **config_dict["embedding_classifier_config"])
        config = BrandEmbeddingClassifierConfig(embedding_clf_config, **config_dict["brand_embedding_classifier_config"])
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = BrandEmbeddingClassifier(config)

    return model

def unicode_clean(s):
    if not isinstance(s, str):
        return s
    
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(c for c in s if unicodedata.category(c)[0] != 'C')

    return s.strip()

def predict_brick_ensemble(
        product_name: str, 
        model,
         exclusion_column = "BrickDefinition_Excludes"
    ) -> str:

        topk_bricks = model.get_gpc(product_name, "brick")

        return model.predict_brick_by_exclusion(
            product_name=product_name,
            candidate_bricks=topk_bricks,
            exclusion_column=exclusion_column
        )

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import unicodedata
from sklearn.metrics import f1_score
from safetensors.torch import save_file

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

def load_tfidf_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config_dict["ngram_range"] = tuple(config_dict["ngram_range"])
    
    try:
        config = TfidfClassifierConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = TfidfClassifier(config)

    return model

def load_embedding_xgb_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        embedding_config = SentenceEmbeddingConfig(**config_dict["embedding_config"])
        config = EmbeddingXGBoostConfig(embedding_config, **config_dict["xgb_config"])
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = EmbeddingXGBoostModel(config)

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

def unicode_clean(s):
    if not isinstance(s, str):
        return s
    
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(c for c in s if unicodedata.category(c)[0] != 'C')

    return s.strip()

def gpc_hierarchical_classifier_train(model: GpcHierarchicalClassifier, x_train , y_train, epochs: int, lr: float):
    model.train()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        xb = torch.tensor(x_train, dtype=model.dtype, device=model.device)
        yb = torch.tensor(y_train, dtype=torch.long, device=model.device)

        optimizer.zero_grad()
        output = model(xb)

        segment_loss = loss_fn(output["segment"], yb[:, 0])
        family_loss = loss_fn(output["family"], yb[:, 1])
        class_loss = loss_fn(output["class"], yb[:, 2])
        brick_loss = loss_fn(output["brick"], yb[:, 3])

        loss = segment_loss + family_loss + class_loss + brick_loss
        loss = segment_loss + family_loss + class_loss

        loss.backward()

        optimizer.step()

        logger.info(f"Epoch {epoch+1}: loss = {loss:.4f}")

    return model

def gpc_hierarchical_classifier_inference(model: GpcHierarchicalClassifier, x: Union[List[float], np.ndarray, torch.Tensor]):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=model.dtype, device=model.device)
    
    with torch.inference_mode():
        logits = model(x)
        segment_prob = torch.softmax(logits["segment"], dim=1)
        family_prob = torch.softmax(logits["family"], dim=1)
        class_prob = torch.softmax(logits["class"], dim=1)
        brick_prob = torch.softmax(logits["brick"], dim=1)
    
    return (
        torch.argmax(segment_prob, dim=1),
        torch.argmax(family_prob, dim=1),
        torch.argmax(class_prob, dim=1),
        torch.argmax(brick_prob, dim=1)
    )

def save_model(model: GpcHierarchicalClassifier) -> None:
    model_path = MODEL_PATH / model.model_name
    state_dict = model.state_dict()

    save_file(state_dict, model_path)
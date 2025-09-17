import unicodedata
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from modules.models import EmbeddingClassifier, EmbeddingClassifierConfig

import re
import json
from typing import List, Union

from constants import RANDOM_STATE
from modules.logger import logger
from modules.models import (
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
    OpusTranslationModel,
    OpusTranslationModelConfig,
    TfidfClassifier, 
    TfidfClassifierConfig,
    EmbeddingSvmConfig,
    EmbeddingSvmModel,
    EmbeddingClassifier,
    EmbeddingClassifierConfig,
    BrandEmbeddingClassifier,
    BrandEmbeddingClassifierConfig,
    EnsembleConfig,
    EnsembleModel,
    TfidfSimilarityConfig,
    TfidfSimilarityModel
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

def clean_(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.strip().split())

def remove_extra_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text) -> str:
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_extra_space(text)

    return text.lower()

def split_dataset(dataset_path: str, train_dataset_path: str, test_dataset_path: str):
    df = pd.read_csv(dataset_path)
    df.dropna(subset=["class"], inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    train_df.to_csv(train_dataset_path, index=False)
    test_df.to_csv(test_dataset_path, index=False)

def load_tfidf_classifier_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config_dict["ngram_range"] = tuple(config_dict["ngram_range"])
    
    try:
        config = TfidfClassifierConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = TfidfClassifier(config)

    return model

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

def load_embedding_svm_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        embedding_config = SentenceEmbeddingConfig(**config_dict["embedding_config"])
        config = EmbeddingSvmConfig(embedding_config, **config_dict["classification_config"])
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = EmbeddingSvmModel(config)

    return model

def load_ensemble_pipeline(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        embedding_config = SentenceEmbeddingConfig(**config_dict["embedding_classifier"]["embedding_config"])
        embed_clf_config = EmbeddingClassifierConfig(embedding_config, **config_dict["embedding_classifier"]["classification_config"])
        brand_tfidf_similiraity_config = TfidfSimilarityConfig(**config_dict["brand_tfidf_similiraity"])
        tfidf_clf_config = TfidfClassifierConfig(**config_dict["tfidf_classifier"])
        config = EnsembleConfig(embed_clf_config, brand_tfidf_similiraity_config, tfidf_clf_config)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = EnsembleModel(config)

    return model

def load_tfidf_similarity_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config_dict["ngram_range"] = tuple(config_dict["ngram_range"])

    try:
        config = TfidfSimilarityConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = TfidfSimilarityModel(config)

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

import matplotlib.pyplot as plt
import pandas as pd

def plot_classification_results(df, level: str):
    df = df.copy()
    col_true = level
    col_pred = f"pred_{level}"

    df["correct"] = df[col_true] == df[col_pred]

    counts = (
        df.groupby([col_true, "correct"])
        .size()
        .unstack(fill_value=0)
    )

    colors = {False: "red", True: "green"}
    ax = counts.plot(
        kind="bar",
        stacked=False,   # side-by-side instead of stacked
        figsize=(14, 6),
        color=[colors[c] for c in counts.columns]
    )

    
    ax.set_ylabel("Number of Predictions (log scale)")
    ax.set_title(f"Distribution of Correctly Classified {level.title()}s")
    ax.legend(["Incorrect", "Correct"], title="Prediction")
    ax.set_yscale("log")

    if level == "segment":
        plt.xticks(rotation=45, ha="right")
        for p, label in zip(ax.patches, [*counts.columns] * len(counts)):
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height * 1.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="green" if label else "red",
                    weight="bold",
                )
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")



    correct_counts = counts.get(True, pd.Series(dtype=int)).sort_values(ascending=False)
    top5 = correct_counts.head(5)
    bottom5 = correct_counts.tail(5)

    textstr = "Top 5 Correctly Classified:\n"
    for idx, val in top5.items():
        textstr += f"{idx}: {val}\n"

    textstr += "\nLowest 5 Correctly Classified:\n"
    for idx, val in bottom5.items():
        textstr += f"{idx}: {val}\n"

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(
        1.05,
        0.5,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=props,
        color="black"
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()
    plt.show()

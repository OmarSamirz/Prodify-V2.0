from sklearn.feature_extraction.text import TfidfVectorizer

import os
import ast
from typing import Any
from abc import ABC, abstractmethod


class BaseTfidfModel(ABC):

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

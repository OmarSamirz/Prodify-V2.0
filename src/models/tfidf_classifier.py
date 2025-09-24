import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

import os
from typing import List, Union
from typing_extensions import override

from constants import ARTIFACTS_PATH
from modules.logger import logger
from models.base_tfidf_model import BaseTfidfModel

load_dotenv()


class TfidfClassifier(BaseTfidfModel):

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
        if os.path.exists(ARTIFACTS_PATH / self.model_name):
            self.load()

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

        logger.info("Predicting `segments`, `families` and `classes` for invoice(s) in `TF-IDF Classifer`.")
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
        if not os.path.exists(ARTIFACTS_PATH):
            os.makedirs(ARTIFACTS_PATH, exist_ok=True)

        model_path = ARTIFACTS_PATH / self.model_name
        joblib.dump(self.clf, model_path)

    @override
    def load(self) -> None:
        if self.clf is not None:
            return

        model_path = ARTIFACTS_PATH / self.model_name
        self.clf = joblib.load(model_path)
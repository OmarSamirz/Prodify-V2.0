import pandas as pd
from sklearn.metrics import accuracy_score

from modules.logger import logger
from models.base_tfidf_model import BaseTfidfModel
from models import BrandsClassifier, TfidfClassifier

def train_tfidf_model(
    tfidf_model: BaseTfidfModel, 
    df_train: pd.DataFrame,
) -> None:
    df_train["product_name"] = df_train["product_name"].astype(str)

    X_train = df_train["product_name"].tolist()
    y_train = df_train["class"].tolist()

    logger.info("Training the model")
    if isinstance(tfidf_model, BrandsClassifier):
        tfidf_model.fit()
    elif isinstance(tfidf_model, TfidfClassifier):
        tfidf_model.fit(X_train, y_train)

    tfidf_model.save()

def test_tfidf_model(
        tfidf_model: BaseTfidfModel,
        df_test: pd.DataFrame
) -> None:
    df_test["product_name"] = df_test["product_name"].astype(str)

    X_test = df_test["product_name"].tolist()
    true_seg = df_test["segment"].tolist()
    true_fam = df_test["family"].tolist()
    true_cls = df_test["class"].tolist()

    y_pred = tfidf_model.predict(X_test)

    logger.info(f"Segment: {accuracy_score(true_seg, y_pred[0])}")
    logger.info(f"Family: {accuracy_score(true_fam, y_pred[1])}")
    logger.info(f"Class: {accuracy_score(true_cls, y_pred[2])}")
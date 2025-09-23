import pandas as pd
from sklearn.metrics import accuracy_score

from typing import Optional

from modules.logger import logger
from modules.models import TfidfBaseModel, BrandsClassifier, TfidfClassifier

def train_tfidf_models(
    tfidf_model: TfidfBaseModel, 
    train_dataset_path: str,
    test_dataset_path: Optional[str] = None
) -> None:
    df_train = pd.read_csv(train_dataset_path)
    df_train["product_name"] = df_train["product_name"].astype(str)

    X_train = df_train["product_name"].tolist()
    y_train = df_train["class"].tolist()

    logger.info("Training the model")
    if isinstance(tfidf_model, BrandsClassifier):
        tfidf_model.fit()
    elif isinstance(tfidf_model, TfidfClassifier):
        tfidf_model.fit(X_train, y_train)

    df_test = pd.read_csv(test_dataset_path)
    df_test["product_name"] = df_test["product_name"].astype(str)

    X_test = df_test["product_name"].tolist()
    true_seg = df_test["segment"].tolist()
    true_fam = df_test["family"].tolist()
    true_cls = df_test["class"].tolist()

    y_pred = tfidf_model.predict(X_test)

    logger.info(f"Class: {accuracy_score(true_cls, y_pred[2])}")
    logger.info(f"Family: {accuracy_score(true_fam, y_pred[1])}")
    logger.info(f"Segment: {accuracy_score(true_seg, y_pred[0])}")

    tfidf_model.save()

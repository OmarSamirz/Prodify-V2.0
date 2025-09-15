import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

import re

from modules.logger import logger
from utils import (
    load_embedding_classifier_model,
    load_brand_embedding_classifier_model,
    load_ensemble_pipeline,
    load_tfidf_similarity_model
)
from constants import (
    FULL_DATASET_PATH,
    FULL_TRAIN_DATASET_PATH,
    FULL_TEST_DATASET_PATH,
    EMBEDDING_CLASSIFIER_CONFIG_PATH,
    BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH,
    FULL_BRAND_OUTPUT_DATASET_PATH,
    FULL_EMBEDDING_MODEL_OUTPUT_DATASET_PATH,
    FULL_ENSEMBLE_MODEL_OUTPUT_DATASET_PATH,
    ENSEMBLE_CONFIG_PATH,
    DETAILED_BRANDS_DATASET_PATH,
    TFIDF_SIMILARITY_CONFIG_PATH
)

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.strip().split())

def test_ensemble():
    pipe = load_ensemble_pipeline(ENSEMBLE_CONFIG_PATH)
    df = pd.read_csv(FULL_TEST_DATASET_PATH)
    df["product_name"] = df["product_name"].astype(str)

    segments = []
    families = []
    classes = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        preds = pipe.run_pipeline(row["product_name"])
        segments.append(clean(preds["segment"]))
        families.append(clean(preds["family"]))
        classes.append(clean(preds["class"]))
    
    df["pred_segment"] = segments
    df["pred_family"] = families
    df["pred_class"] = classes
    df.to_csv(FULL_ENSEMBLE_MODEL_OUTPUT_DATASET_PATH, index=False)

    true_segment = df["segment"].tolist()
    true_family = df["pred_family"].tolist()
    true_class = df["pred_class"].tolist()
    logger.info(f"Level segment: {accuracy_score(true_segment, segments)}")
    logger.info(f"Level family: {accuracy_score(true_family, families)}")
    logger.info(f"Level class: {accuracy_score(true_class, classes)}")

def test_tfidf_similarity_model():
    df = pd.read_csv(DETAILED_BRANDS_DATASET_PATH)
    df["documents"] = df["Sector"] + " " + df["Brand"] + " " + df["Product"]
    documents = df["documents"].tolist()

    model = load_tfidf_similarity_model(TFIDF_SIMILARITY_CONFIG_PATH)
    model.fit(documents)
    model.save()
    predictions = model.find_similarity("Apple", documents)
    logger.info(f"The top {model.topk} predictions: {predictions}")

def test_brand_embedding_model():
    model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    brand_data = model.get_brand_data("Harry Potter and The Chamber of Secrets")
    print(brand_data)

def embedding_classifier_test():
    embed_cls = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    gpc_labels = embed_cls.get_gpc("Apple iphone 12 pro max")
    print(gpc_labels)

def exclusion_test():
    test_products = [
        "Apple MacBook Pro",
        "Harry Potter and the Goblet of Fire Book",
        "Samsung Galaxy S22 Smartphone",
        "Nike Air Max Running Shoes",
        "Colgate Total Toothpaste",
        "Nestle KitKat Chocolate Bar",
        "Sony Bravia 55 Inch TV",
        "Sharp Double Door Refrigerator",
        "Oxy Whitening Face Wash",
        "Chipsy Salted Potato Chips"
    ]

    brand_model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    model = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    with open("data/testing_mixed.txt", "w") as f:
        for pr in test_products:
            if brand_model.get_brand_data(pr):
                f.write(f"Brand Found for Product: {pr}\n")
                gpc_labels = brand_model.get_gpc(pr)
            else:
                gpc_labels= model.get_gpc(pr)
                f.write(f"Brand not Found for Product: {pr}\n")
            for i in gpc_labels:
                f.write(i)
                f.write("\n")
            f.write("\n")

def main():
    test_tfidf_similarity_model()
    # segments = []
    # families = []
    # classes = []
    # df = pd.read_csv(FULL_TEST_DATASET_PATH)
    # brand_model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    # model = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    # df["product_name"] = df["product_name"].astype(str)
    # for _, row in tqdm(df.iterrows(), total=len(df)):
    #     pr = row["product_name"]
    #     gpc_labels = model.get_gpc(pr)
    #     segments.append(gpc_labels[0])
    #     families.append(gpc_labels[1])
    #     classes.append(gpc_labels[2])

    # df["pred_segment"] = segments
    # df["pred_family"] = families
    # df["pred_class"] = classes
    # df.to_csv(FULL_EMBEDDING_MODEL_OUTPUT_DATASET_PATH, index=False)

    # true_segment = df["segment"].tolist()
    # true_family = df["pred_family"].tolist()
    # true_class = df["pred_class"].tolist()
    # logger.info(f"Level segment: {accuracy_score(true_segment, segments)}")
    # logger.info(f"Level family: {accuracy_score(true_family, families)}")
    # logger.info(f"Level class: {accuracy_score(true_class, classes)}")


if __name__ == "__main__":
    main()
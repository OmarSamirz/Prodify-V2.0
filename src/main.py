import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from modules.logger import logger
from utils import load_embedding_classifier_model, load_brand_embedding_classifier_model, split_dataset
from constants import (
    FULL_DATASET_PATH,
    FULL_TRAIN_DATASET_PATH,
    FULL_TEST_DATASET_PATH,
    EMBEDDING_CLASSIFIER_CONFIG_PATH,
    BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH,
    FINAL_DB,
    FULL_OUTPUT_DATASET_PATH
)

def test_brand_embedding_model():
    model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    brand_data = model.get_brand_data("Harry Potter and The Chamber of Secrets")
    print(brand_data)

def embedding_classifier_test():
    embed_cls = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    gpc_labels = embed_cls.get_gpc("Apple iphone 12 pro max")
    print(gpc_labels)

def load_data_set_from_db():
    #tdf = td_db.execute_query("Select * from demo_user.full_dataset")
    df = pd.read_csv(FINAL_DB)
    return df

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
    segments = []
    families = []
    classes = []
    df = pd.read_csv(FULL_TEST_DATASET_PATH)
    brand_model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    model = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pr = row["product_name"]
        if brand_model.get_brand_data(pr):
            gpc_labels = brand_model.get_gpc(pr)
            segments.append(gpc_labels[0])
            families.append(gpc_labels[1])
            classes.append(gpc_labels[2])
        else:
            gpc_labels = model.get_gpc(pr)
            segments.append(gpc_labels[0])
            families.append(gpc_labels[1])
            classes.append(gpc_labels[2])
    
    df["pred_segment"] = segments
    df["pred_family"] = families
    df["pred_class"] = classes
    df.to_csv(FULL_OUTPUT_DATASET_PATH, index=False)

    true_segment = df["segment"].tolist()
    true_family = df["pred_family"].tolist()
    true_class = df["pred_class"].tolist()
    logger.info(f"Level segment: {accuracy_score(true_segment, segments)}")
    logger.info(f"Level family: {accuracy_score(true_family, families)}")
    logger.info(f"Level class: {accuracy_score(true_class, classes)}")


if __name__ == "__main__":
    main()
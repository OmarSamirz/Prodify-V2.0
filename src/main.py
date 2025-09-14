import pandas as pd

from utils import load_embedding_classifier_model, load_brand_embedding_classifier_model, split_dataset
from constants import (
    FULL_DATASET_PATH,
    FULL_TRAIN_DATASET_PATH,
    FULL_TEST_DATASET_PATH,
    EMBEDDING_CLASSIFIER_CONFIG_PATH,
    BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH,
)

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
        "Apple MacBook",
        "Harry Potter and The Chamber of Secrets",
        "Sharp Fridge",
        "Oxy Hand powder",
        "Chipsy Tomato"
    ]

    model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    with open("testing.txt", "w") as f:
        for pr in test_products:
            gpc_labels = model.get_gpc(pr)
            f.write(f"Product: {pr}\n")
            for i in gpc_labels:
                f.write(i)
                f.write("\n")
            f.write("\n")

def main():
    split_dataset(
        FULL_DATASET_PATH,
        FULL_TRAIN_DATASET_PATH,
        FULL_TEST_DATASET_PATH
    )

if __name__ == "__main__":
    main()
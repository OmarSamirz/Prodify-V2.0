import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from modules.logger import logger
from modules.db import TeradataDatabase
from utils import load_embedding_classifier_model, load_brand_embedding_classifier_model
from constants import (
    EMBEDDING_CLASSIFIER_CONFIG_PATH,
    BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH,
    FINAL_DB
)


td_db = TeradataDatabase() 
td_db.connect()
RANDOM_SEED = 42

hierarchy = ['segment','family','class','brick']

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
    df_all = load_data_set_from_db()
    te = df_all.sample(frac=0.1, random_state=42)
    brand_model = load_brand_embedding_classifier_model(BRAND_EMBEDDING_CLASSIFIER_CONFIG_PATH)
    model = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    accuracies = {}
    for layer in tqdm(hierarchy, total=len(hierarchy)):
        if te.empty:
            continue
        y_true, y_pred = [], []

        for _, row in te.iterrows():
            product = row['product_name']

            if brand_model.get_brand_data(product):
                gpc_labels = brand_model.get_gpc(product)
            else:
                gpc_labels = model.get_gpc(product)

            label_map = dict(zip(hierarchy, gpc_labels))
            if layer in label_map:
                    y_pred.append(label_map[layer])
                    y_true.append(row[layer])

        acc = accuracy_score(y_true, y_pred)
        accuracies[layer] = acc
        logger.info(f"[{layer.upper()}] Accuracy: {acc:.4f}")

    # print("\nOverall Results:")
    # for l, acc in accuracies.items():
    #     print(f"{l.capitalize()}: {acc:.4f}")

if __name__ == "__main__":
    main()
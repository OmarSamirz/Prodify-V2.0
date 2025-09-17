import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

import re

from modules.logger import logger
from utils import (
    load_embedding_classifier_model,
    load_brand_embedding_classifier_model,
    load_ensemble_pipeline,
    load_tfidf_similarity_model,
    load_tfidf_classifier_model
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
    TFIDF_SIMILARITY_CONFIG_PATH,
    TFIDF_CLASSIFIER_CONFIG_PATH,
    CLASS_ONLY_CLASSIFIER,
    FULL_TFIDF_SIMILARITY_OUTPUT_DATASET_PATH
)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from safetensors.torch import save_file, load_file

from utils import (
    load_gpc_hierarchical_classifier, 
    gpc_hierarchical_classifier_train,
    gpc_hierarchical_classifier_inference, 
    load_embedding_model
)
from constants import (
    GPC_HIERARCHICAL_CLASSIFIER_CONFIG, 
    FULL_TRAIN_DATASET_PATH, 
    FULL_TEST_DATASET_PATH, 
    E5_LARGE_INSTRUCT_CONFIG_PATH
)

def test_gpc_model():
    df_train = pd.read_csv(FULL_TRAIN_DATASET_PATH)
    df_test = pd.read_csv(FULL_TEST_DATASET_PATH)
    df_merged = pd.concat([df_train, df_test])
    
    seg_encoder = LabelEncoder()
    fam_encoder = LabelEncoder()
    cls_encoder = LabelEncoder()
    df_merged["encoded_segment"] = seg_encoder.fit_transform(df_merged["segment"].tolist())
    df_merged["encoded_family"] = fam_encoder.fit_transform(df_merged["family"].tolist())
    df_merged["encoded_class"] = cls_encoder.fit_transform(df_merged["class"].tolist())

    df_train_len = len(df_train)
    df_train = df_merged.iloc[:df_train_len, :]
    df_test = df_merged.iloc[df_train_len:, :]
    X_train, y_train = df_train["product_name"].astype(str).tolist(), df_train[["encoded_segment", "encoded_family", "encoded_class"]].values.tolist()
    X_test, y_test = df_test["product_name"].astype(str).tolist(), df_test[["encoded_segment", "encoded_family", "encoded_class"]].values.tolist()
    
    X_train = load_file("src/train_embeddings.safetensors")["input"]
    X_test = load_file("src/test_embeddings.safetensors")["input"]

    gpc_model = load_gpc_hierarchical_classifier(GPC_HIERARCHICAL_CLASSIFIER_CONFIG).to("cuda")
    model, best_state = gpc_hierarchical_classifier_train(
        model=gpc_model,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        epochs=500,
        lr=0.01
    )
    logger.info(f"The best epoch is {best_state["epoch"]}")

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.strip().split())

def test_ensemble():
    pipe = load_ensemble_pipeline(ENSEMBLE_CONFIG_PATH)
    # pred = pipe.run_pipeline("Nike Air Max Running Shoes")
    # logger.info(f"Level segment: {pred}")

    df = pd.read_csv(FULL_TEST_DATASET_PATH)
    df["product_name"] = df["product_name"].astype(str)

    segments = []
    families = []
    classes = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        preds = pipe.run_pipeline(row["product_name"])
        segments.append(preds["segment"])
        families.append(preds["family"])
        classes.append(preds["class"])

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

def test_classifier():
    model = load_tfidf_classifier_model(TFIDF_CLASSIFIER_CONFIG_PATH)
    df_train = pd.read_csv(FULL_TRAIN_DATASET_PATH)
    df_test = pd.read_csv(FULL_TEST_DATASET_PATH)
    # df["product_name"] = df["product_name"].astype(str)
    X_train = df_train["product_name"].fillna("").astype(str).tolist()
    y_train = df_train["class"].fillna("").astype(str).tolist()

    X_test = df_test["product_name"].fillna("").astype(str).tolist()

    model.fit(X_train, y_train)
    model.save()

    X_test = df_test["product_name"].fillna("").astype(str).tolist()
    y_pred = model.predict(X_test)

    class_to_segment = df_train.set_index("class")["segment"].to_dict()
    class_to_family = df_train.set_index("class")["family"].to_dict()

    pred_segment = [class_to_segment.get(c, "") for c in y_pred]
    pred_family = [class_to_family.get(c, "") for c in y_pred]

    df_test["pred_segment"] = pred_segment
    df_test["pred_family"] = pred_family
    df_test["pred_class"] = y_pred

    df_test.to_csv(CLASS_ONLY_CLASSIFIER, index=False)

    true_segment = df_test["segment"].tolist()
    true_family = df_test["family"].tolist()
    true_class = df_test["class"].tolist()

    logger.info(f"Level segment accuracy: {accuracy_score(true_segment, pred_segment)}")
    logger.info(f"Level family accuracy: {accuracy_score(true_family, pred_family)}")
    logger.info(f"Level class accuracy: {accuracy_score(true_class, y_pred)}")

def test_tfidf_similarity_model():
    df = pd.read_csv(DETAILED_BRANDS_DATASET_PATH)
    df["documents"] = df["Sector"] + " " + df["Brand"] + " " + df["Product"]
    documents = df["documents"].tolist()
    df["documents"] = df["documents"].apply(clean)
    
    df_test = pd.read_csv(FULL_TEST_DATASET_PATH)
    X_test = df_test["product_name"].fillna("").astype(str).apply(clean).tolist()

    # df["target_label"] =  df["Segment"] + " " + df["Family"] + " " + df["Class"]
    model = load_tfidf_similarity_model(TFIDF_SIMILARITY_CONFIG_PATH)
    model.fit(documents)
    model.save()

    pred_segments, pred_families, pred_classes = [], [], []

    for product in tqdm(X_test, total=len(X_test)):
        indices = model.find_similarity(product, documents)
        top_row = df.iloc[indices[0]]
        pred_segments.append(top_row["Segment"])
        pred_families.append(top_row["Family"])
        pred_classes.append(top_row["Class"])

    true_segments = df_test["segment"].tolist()
    true_families = df_test["family"].tolist()
    true_classes = df_test["class"].tolist()

    df_test["pred_segment"] = pred_segments
    df_test["pred_family"] = pred_families
    df_test["pred_class"] = pred_classes

    df_test.to_csv(FULL_TFIDF_SIMILARITY_OUTPUT_DATASET_PATH, index=False)

    logger.info(f"BRAND MODEL FROM HERE! {accuracy_score(true_segments, pred_segments)}")
    logger.info(f"Level segment accuracy: {accuracy_score(true_segments, pred_segments)}")
    logger.info(f"Level family accuracy: {accuracy_score(true_families, pred_families)}")
    logger.info(f"Level class accuracy: {accuracy_score(true_classes, pred_classes)}")

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
    test_ensemble()
    #test_tfidf_similarity_model()

    # Run embedding model
    # segments = []
    # families = []
    # classes = []
    # df = pd.read_csv(FULL_TEST_DATASET_PATH)

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
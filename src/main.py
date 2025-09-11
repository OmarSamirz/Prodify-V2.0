from utils import load_embedding_classifier_model, load_brand_embedding_classifier_model

from constants import (
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

    model = load_embedding_classifier_model(EMBEDDING_CLASSIFIER_CONFIG_PATH)
    with open("testing.txt", "w") as f:
        for pr in test_products:
            gpc_labels = model.get_gpc(pr)
            best_brick = model.predict_brick_by_exclusion(pr, gpc_labels[-3:])
            ls = gpc_labels[:3]
            ls.append(best_brick)
            f.write(f"Product: {pr}\n")
            for i in ls:
                f.write(i)
                f.write("\n")
            f.write("\n")

def main():
    test_brand_embedding_model()

if __name__ == "__main__":
    main()

from utils import load_embedding_classifier_model, predict_brick_ensemble

from pipelines import AmurdPipeline, GpcPipeline

from constants import (
    GPC_PATH,
    TRAIN_VAL_DATA_PATH,
    TEST_DATA_PATH,
    E5_LARGE_INSTRUCT_CONFIG_PATH,
    OPUS_TRANSLATION_CONFIG_PATH,
    TFIDF_CLASSIFIER_CONFIG_PATH,
    EMBEDDING_CLASSIFIER_CONFIG_PATH
)

hierarchy = ['segment','family','class','brick']

def amurd_pipeline():
    pipe = AmurdPipeline(
        df_train_path=TRAIN_VAL_DATA_PATH,
        df_test_path=TEST_DATA_PATH,
        embedding_model_config_path=E5_LARGE_INSTRUCT_CONFIG_PATH,
        translation_model_config_path=OPUS_TRANSLATION_CONFIG_PATH,
        tfidf_classifier_config_path=TFIDF_CLASSIFIER_CONFIG_PATH
    )
    pipe.run_pipeline()

def gpc_pipeline():
    pipe = GpcPipeline(
        df_gpc_path=GPC_PATH,
        df_train_path=TRAIN_VAL_DATA_PATH,
        df_test_path=TEST_DATA_PATH,
        embedding_model_config_path=E5_LARGE_INSTRUCT_CONFIG_PATH,
        translation_model_config_path=OPUS_TRANSLATION_CONFIG_PATH,
        tfidf_classifier_config_path=TFIDF_CLASSIFIER_CONFIG_PATH
    )
    pipe.run_pipeline()

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
    # for product in test_products:
    #     for level in hierarchy:
    #         if level != "brick":
    #             pred = model.get_gpc(product, level)
    #             print(f"[{product}] → {level} → {pred}")
    #         else:
    #             pred = predict_brick_ensemble(product, model)
    #             print(f"[{product}] → {level} → {pred}")

def main():
    exclusion_test()

if __name__ == "__main__":
    main()
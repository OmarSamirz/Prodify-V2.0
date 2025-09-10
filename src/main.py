
from utils import load_embedding_classifier_model
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

def main():
    embedding_classifier_test()


if __name__ == "__main__":
    main()
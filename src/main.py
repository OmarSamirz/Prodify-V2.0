
from pipelines import AmurdPipeline, GpcPipeline
from constants import (
    GPC_PATH,
    TRAIN_VAL_DATA_PATH,
    TEST_DATA_PATH,
    E5_LARGE_INSTRUCT_CONFIG_PATH,
    OPUS_TRANSLATION_CONFIG_PATH,
    TFIDF_CLASSIFIER_CONFIG_PATH
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

def main():
    amurd_pipeline()


if __name__ == "__main__":
    main()
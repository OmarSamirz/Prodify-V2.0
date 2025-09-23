from modules.logger import logger
from pipelines import ProdifyPipeline

from constants import FULL_TRAIN_DATASET_PATH, FULL_TEST_DATASET_PATH

def test_pipeline():
    invoice_items = "Apple MacBook"
    pipe = ProdifyPipeline(df_train_path=FULL_TRAIN_DATASET_PATH, df_test_path=FULL_TEST_DATASET_PATH)
    pipe.run_pipeline()
    predicitons = pipe.run_inference(invoice_items)
    logger.info(predicitons)

def main():
    test_pipeline()


if __name__ == "__main__":
    main()
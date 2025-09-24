from pipelines import EnsemblePipeline
from constants import FULL_TRAIN_DATASET_PATH, FULL_TEST_DATASET_PATH

def run_pipeline():
    pipe = EnsemblePipeline(df_train_path=FULL_TRAIN_DATASET_PATH, df_test_path=FULL_TEST_DATASET_PATH)
    pipe.run_pipeline()

def main():
    run_pipeline()


if __name__ == "__main__":
    main()
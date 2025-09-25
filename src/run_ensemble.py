import pandas as pd

from pipelines import EnsemblePipeline
from constants import (
    TRAIN_DATASET_PATH, 
    TEST_DATASET_PATH, 
    AMURD_TRANSLATED_DATASET_PATH, 
    ENSEMBLE_PIPELINE_OUTPUT_PATH
)

def run_pipeline():
    pipe = EnsemblePipeline(df_train_path=TRAIN_DATASET_PATH, df_test_path=TEST_DATASET_PATH)
    pipe.run_pipeline()

def run_inference_on_amurd():
    amurd_df = pd.read_csv(AMURD_TRANSLATED_DATASET_PATH)
    products = amurd_df["product_name"].tolist()

    pipe = EnsemblePipeline()
    _ = pipe.run_inference("صباح الخير")

    pipe_df = pd.read_csv(ENSEMBLE_PIPELINE_OUTPUT_PATH)
    df_merged = pd.merge(amurd_df, pipe_df, on="product_name", how="left")
    df_merged.drop_duplicates(subset=["product_name"], inplace=True)
    df_merged.to_csv(ENSEMBLE_PIPELINE_OUTPUT_PATH, index=False)


def main():
    run_inference_on_amurd()


if __name__ == "__main__":
    main()
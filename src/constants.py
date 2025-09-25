import torch

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_STATE = 42

BASE_DIR = Path(__file__).parents[1]

ANALYSIS_DIR = BASE_DIR / "analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH = ANALYSIS_DIR / "full_confidence_distribution.jpg"

CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH = ANALYSIS_DIR / "correct_confidence_distribution.jpg"

INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH = ANALYSIS_DIR / "incorrect_confidence_distribution.jpg"

MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH = ANALYSIS_DIR / "model_performance_segment.jpg"

MODEL_PERFORMANCE_FAMILY_GRAPH_PATH = ANALYSIS_DIR / "model_performance_family.jpg"

MODEL_PERFORMANCE_CLASS_GRAPH_PATH = ANALYSIS_DIR / "model_performance_class.jpg"

DATA_PATH = BASE_DIR / "data"

MODEL_OUTPUTS_DATA_PATH = DATA_PATH / "outputs"

ORIGINAL_DATASETS_PATH = DATA_PATH / "original_datasets"

IMG_PATH = BASE_DIR / "img"

DATASET_PATH = DATA_PATH / "dataset.csv"

TRAIN_DATASET_PATH = DATA_PATH / "train_dataset.csv"

BRANDS_DATASET_PATH = DATA_PATH / "brands_dataset.csv"

TEST_DATASET_PATH = DATA_PATH / "test_dataset.csv"

GPC_PATH = DATA_PATH / "gpc.csv"

AMURD_TRANSLATED_DATASET_PATH = DATA_PATH / "amurd_translated_dataset.csv"

MWPD_FULL_DATASET_PATH = ORIGINAL_DATASETS_PATH / "mwpd_full_dataset.csv"

USDA_FULL_DATASET_PATH = ORIGINAL_DATASETS_PATH / "usda_full_dataset.csv"

JIO_MART_DATASET_PATH = ORIGINAL_DATASETS_PATH / "jio_mart_dataset.csv"

BRAND_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "brand_output_dataset.csv"

EMBEDDING_MODEL_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "embedding_model_output_dataset.csv"

TFIDF_SIMILARITY_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "tfidf_similarity_output_dataset.csv"

ENSEMBLE_MODEL_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "ensemble_model_output_dataset.csv"

ENSEMBLE_PIPELINE_OUTPUT_PATH = MODEL_OUTPUTS_DATA_PATH / "ensemble_pipeline_output.csv"

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"

ARTIFACTS_PATH = BASE_DIR / "artifacts"

TRANSLATION_MODEL_PATH = ARTIFACTS_PATH / "translation_model"

EMBEDDING_MODEL_PATH = ARTIFACTS_PATH / "embedding_model"

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
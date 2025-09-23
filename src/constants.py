import torch

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_STATE = 42

BASE_DIR = Path(__file__).parents[1]

GRAPHS_DIR = BASE_DIR / "analysis"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

FULL_CONFIDENCE_DISTRIBUTION_GRAPH_PATH = GRAPHS_DIR / "full_confidence_distribution.jpg"

CORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH = GRAPHS_DIR / "correct_confidence_distribution.jpg"

INCORRECT_CONFIDENCE_DISTRIBUTION_GRAPH_PATH = GRAPHS_DIR / "incorrect_confidence_distribution.jpg"

MODEL_PERFORMANCE_SEGMENT_GRAPH_PATH = GRAPHS_DIR / "model_performance_segment.jpg"

MODEL_PERFORMANCE_FAMILY_GRAPH_PATH = GRAPHS_DIR / "model_performance_family.jpg"

MODEL_PERFORMANCE_CLASS_GRAPH_PATH = GRAPHS_DIR / "model_performance_class.jpg"

MODEL_PERFORMANCE_SEGMENT_BY_FAMILIES_BASE_PATH = GRAPHS_DIR / "model_performance_segment_by_family.jpg"

MODEL_PERFORMANCE_FAMILY_BY_CLASSES_BASE_PATH = GRAPHS_DIR / "model_performance_family_by_class.jpg"

DATA_PATH = BASE_DIR / "data"

MODEL_OUTPUTS_DATA_PATH = DATA_PATH / "outputs"

ORIGINAL_DATASETS_PATH = DATA_PATH / "original_datasets"

IMG_PATH = BASE_DIR / "img"

FULL_DATASET_PATH = DATA_PATH / "full_dataset.csv"

FULL_TEST_DATASET_PATH = ORIGINAL_DATASETS_PATH / "full_test_dataset.csv"

FULL_TRAIN_DATASET_PATH = ORIGINAL_DATASETS_PATH / "full_train_dataset.csv"

MWPD_FULL_DATASET_PATH = ORIGINAL_DATASETS_PATH / "mwpd_full_dataset.csv"

USDA_FULL_DATASET_PATH = ORIGINAL_DATASETS_PATH / "usda_full_dataset.csv"

FULL_BRAND_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "full_brand_output_dataset.csv"

FULL_EMBEDDING_MODEL_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "full_embedding_model_output_dataset.csv"

FULL_TFIDF_SIMILARITY_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "full_tfidf_similarity_output_dataset.csv"

FULL_ENSEMBLE_MODEL_OUTPUT_DATASET_PATH = MODEL_OUTPUTS_DATA_PATH / "full_ensemble_model_output_dataset.csv"

CLASS_ONLY_CLASSIFIER = DATA_PATH / "full_tfidf_svm_output_dataset.csv"

BRANDS_DATASET_PATH = DATA_PATH / "brands_dataset.csv"

GPC_PATH = DATA_PATH / "gpc.csv"

JIO_MART_DATASET_PATH = DATA_PATH / "jio_mart_dataset.csv"

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"

MODEL_PATH = BASE_DIR / "models"

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
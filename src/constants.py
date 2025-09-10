import torch

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_STATE = 42

BASE_DIR = Path(__file__).parents[1]

IMG_PATH = BASE_DIR / "img"

ASSETS_PATH = BASE_DIR / "assets"

DATA_PATH = BASE_DIR / "data"

SEOUDI_DATA_PATH = BASE_DIR / DATA_PATH / "seoudi"

SEOUDI_DATA_PATH.mkdir(parents=True, exist_ok=True)

SEOUDI_EN_DATA_PATH = SEOUDI_DATA_PATH / "seoudi_products_en.csv"

SEOUDI_AR_DATA_PATH = SEOUDI_DATA_PATH / "seoudi_products_ar.csv"

CARREFOUR_DATA_PATH = BASE_DIR / DATA_PATH / "carrefour"

CARREFOUR_DATA_PATH.mkdir(parents=True, exist_ok=True)

CARREFOUR_CSV_DATA_PATH = CARREFOUR_DATA_PATH / "carrefour.csv"

TRAIN_VAL_DATA_PATH = DATA_PATH / "train_val.csv"

TRAIN_CLEANED_DATA_PATH = DATA_PATH / "train_cleaned.csv"

VALIDATION_DATA_PATH = DATA_PATH / "validation.csv"

TEST_DATA_PATH = DATA_PATH / "test.csv"

FULL_DATASET_PATH = DATA_PATH / "full_dataset.csv"

CLEANED_TEST_DATA_PATH = DATA_PATH / "cleaned_test.csv"

CLEANED_TRAIN_DATA_PATH = DATA_PATH / "cleaned_train.csv"

CLEANED_FULL_DATASET_PATH = DATA_PATH / "cleaned_full_dataset.csv"

CLEANED_PRODUCTS_PATH = DATA_PATH / "cleaned_products.csv"

CLEANED_ACTUAL_CLASSES_PATH = DATA_PATH / "cleaned_actual_classes.csv"

CLEANED_CLASSES_PATH = DATA_PATH / "cleaned_classes.csv"

GPC_PATH = DATA_PATH / "gpc.xlsx"

GPC_TRAIN_PATH = DATA_PATH / "gpc_train.csv"

CLEANED_GPC_PATH = DATA_PATH / "cleaned_gpc.csv"

LABELED_GPC_PATH = DATA_PATH / "gpc_labeled.csv"

LABELED_GPC_PRED_PATH = DATA_PATH / "gpc_labeled_pred.csv"

SAMPLE_PRODUCTS_PATH = DATA_PATH / "products_sample.csv"

LABELED_PRODUCTS_PATH = DATA_PATH / "labeled_products.csv"

PRODUCT_TEST_EMBEDDINGS_PATH = DATA_PATH / "product_test_embeddings.csv"

PRODUCT_TRAIN_EMBEDDINGS_PATH = DATA_PATH / "product_train_embeddings.csv"

PRODUCT_FULL_DATASET_EMBEDDINGS_PATH = DATA_PATH / "product_full_dataset_embeddings.csv"

PRODUCT_FULL_DATASET_EMBEDDINGS_QWEN_PATH  = DATA_PATH / "product_full_dataset_embeddings_qwen.csv"

CLASS_EMBEDDINGS_PATH = DATA_PATH / "class_embeddings.csv"

CLASS_EMBEDDINGS_PATH_QWEN = DATA_PATH / "class_embeddings_qwen.csv"

CONFIG_PATH = BASE_DIR / "config"

E5_LARGE_INSTRUCT_CONFIG_PATH = CONFIG_PATH / "e5_large_instruct_config.json"

OPUS_TRANSLATION_CONFIG_PATH = CONFIG_PATH / "opus_translation_config.json"

TFIDF_CLASSIFIER_CONFIG_PATH = CONFIG_PATH / "tfidf_classifier_config.json"

EMBEDDING_XGB_CONFIG_PATH = CONFIG_PATH / "embedding_xgb_config.json"

GPC_HIERARCHICAL_CLASSIFIER_CONFIG = CONFIG_PATH / "gpc_hierarchical_classifier_config.json"

ENV_PATH = CONFIG_PATH / ".env"

MODEL_PATH = BASE_DIR / "models"

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

SEOUDI_QUERY = """
query Products($page:Int $pageSize:Int $search:String $filter:ProductAttributeFilterInput = {} $sort:ProductAttributeSortInput = {}) {
  connection: products(
    currentPage: $page
    pageSize: $pageSize
    filter: $filter
    search: $search
    sort: $sort
  ) {
    total_count
    page_info {
      total_pages
      current_page
      page_size
    }
    nodes: items {
      id
      name
      sku
      url_key
      brand { name }
      categories { url_path name level}
      thumbnail { url }
      price_range {
        maximum_price {
          final_price { value }
          regular_price { value }
        }
      }
    }
  }
}
"""

COUTNRY_LOCATION = {
    "Pakistan": {
        "latitude": "33.69649529437605",
        "longitude": "73.04532569424234",
    },
    'United Arab Emirates': {
        "latitude": "24.426024389026107",
        "longitude": "54.428812748171495",
    },
    'Saudi Arabia': {
        "latitude": "24.704986696953327",
        "longitude": "46.68406531065149",
    },
    'Egypt': {
        "latitude": "30.012047628678435",
        "longitude": "31.422947054156275",
    },
    'Qatar': {
        "latitude": "25.27146789795238",
        "longitude": "51.51973339021754",
    },
    'Lebanon': {
        "latitude": "33.87505039209283",
        "longitude": "35.50283238664546",
    },
    'Kuwait': {
        "latitude": "29.36370888276018",
        "longitude": "47.977422085098155",
    },
    'Kenya': {
        "latitude": "-1.298444534761646",
        "longitude": "36.848596481360616",
    },
    'Georgia': {
        "latitude": "41.712370890152265",
        "longitude": "44.7962471415355",
    },  
}

CARREFOUR_PARAMS = {
  "sortBy": "relevance",
  "categoryCode": "",
  "needFilter": "false",
  "pageSize": 40,
  "requireSponsProducts": "true",
  "verticalCategory": "true",
  "needVariantsData": "true",
  "currentPage": 0,
  "responseWithCatTree": "true",
  "depth": 3,
  "lang": "en",
  "categoryId": "",
  "latitude": 25.2171003,
  "longitude": 55.3613635

}

CARREFOUR_HEADER = {
    "storeid": "mafegy",
    "appid": "Reactweb",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Referer": "https://www.carrefouruae.com/",
    # "channel": "c4online",
    "currency": "EGP",
    "env": "prod",
    "intent": "STANDARD",
    "langcode": "en",
    "posinfo": "",
    "producttype": "ANY",
}
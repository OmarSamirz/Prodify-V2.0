import pandas as pd
from sklearn.metrics import accuracy_score

from typing_extensions import override
from typing import Optional, Dict, List, Any, Union

from modules.logger import logger
from utils import get_confidence_level
from pipelines.base_pipeline import Pipeline
from model_utils import train_tfidf_model, test_tfidf_model
from models import EnsembleModel, BrandsClassifier, EmbeddingClassifier
from constants import ENSEMBLE_MODEL_OUTPUT_DATASET_PATH, ENSEMBLE_PIPELINE_OUTPUT_PATH


class EnsemblePipeline(Pipeline):

    def __init__(
        self,
        df_train_path: Optional[str] = None,
        df_val_path: Optional[str] = None,
        df_test_path: Optional[str] = None,
        is_save_predections: bool = True
    ) -> None:
        super().__init__(
            df_train_path,
            df_val_path,
            df_test_path,
        )
        self.is_save_predections = is_save_predections
        self.embedding_classifier = None
        self.brands_classifier = None
        self.ensemble_model = None

    @override
    def load_dataframe(self, csv_path: str) -> pd.DataFrame:
        return super().load_dataframe(csv_path)

    @override
    def rename_columns(self, dataframe: pd.DataFrame, renamed_columns: Dict[str, str]) -> pd.DataFrame:
        return super().rename_columns(dataframe, renamed_columns)
    
    @override
    def combine_db_table_names(self, table_name: str) -> str:
        return super().combine_db_table_names(table_name)

    @override
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        return super().execute_query(query)

    @override
    def drop_column(self, table_name: str, cols: Union[str, List[str]]) -> None:
        super().drop_column(table_name, cols)

    @override
    def dropna(self, table_name: str, cols: Union[str, List[str]]) -> None:
        super().dropna(table_name, cols)

    @override
    def drop_duplicates(self, table_name: str, cols: Union[str, List[str]], id_name: str = "id") -> None:
        super().drop_duplicates(table_name, cols, id_name)

    @override
    def update_id(self, table_name: str, id_name: str, order_by_col: str, selected_cols: Optional[Union[str, List[str]]] = None) -> None:
        super().update_id(table_name, id_name, order_by_col, selected_cols)

    @override
    def replicate_table(self, current_table_name: str, new_table_name: str, cols: Union[str, List[str]]) -> None:
        super().replicate_table(current_table_name, new_table_name, cols)

    @override
    def drop_table(self, table_name: str) -> None:
        super().drop_table(table_name)

    @override
    def create_table_class(
        self, 
        new_table_name: str,
        current_table_name: str,
        id_col: str, 
        distinct_col_name: str,
        selected_cols: Union[str, List[str]]
    ) -> None:
        super().create_table_class(new_table_name, current_table_name, id_col, distinct_col_name, selected_cols)

    @override
    def get_table(self, table_name: str) -> pd.DataFrame:
        return super().get_table(table_name)

    @override
    def count_table_rows(self, table_name: str, col: str) -> int:
        return super().count_table_rows(table_name, col)
    
    @override
    def database_insertion(
        self, 
        dataframe: pd.DataFrame, 
        table_name: str, 
        if_exists: str = "replace"
    ) -> None:
        super().database_insertion(dataframe, table_name, if_exists)

    @override
    def cleanse_table_column(self, table_name: str, col_name: str) -> None:
        super().cleanse_table_column(table_name, col_name)

    @override
    def concatenate_tables(
        self, 
        table_name: str, 
        selected_cols: Union[str, List[str]], 
    ) -> None:
        super().concatenate_tables(table_name, selected_cols)
    
    @override
    def create_table_from(self, target_table_name: str, source_table_name: str, target_cols: Union[str, List[str]]) -> None:
        super().create_table_from(target_table_name, source_table_name, target_cols)

    @override
    def update_table_columns(
        self,
        target_table: str,
        source_table: str,
        target_col: str,
        source_col: str,
        target_condition_col: Union[str, List[str]],
        source_condition_col: Union[str, List[str]]
    ) -> None:
        super().update_table_columns(target_table, source_table, target_col, source_col, target_condition_col, source_condition_col)

    @override
    def load_embedding_model(self) -> None:
        super().load_embedding_model()

    @override
    def load_tfidf_classifier(self) -> None:
        super().load_tfidf_classifier()

    @override
    def load_translation_model(self) -> None:
        super().load_translation_model()

    @override
    def create_embeddings(
        self,
        table_name: str,
        embedding_col: str,
        new_table_name: str,
        embeddings_name: str = "embed_",
    ) -> None:
        super().create_embeddings(table_name, embedding_col, new_table_name, embeddings_name)

    def load_brands_classifier(self) -> None:
        if self.brands_classifier is not None:
            logger.info("The brands classifier model is loaded.")
            return

        logger.info("Loading brands classifier model.")
        self.brands_classifier = BrandsClassifier()
        logger.info("Loading brands classifier model is done.")

    def load_embedding_classifier(self) -> None:
        if self.embedding_classifier is not None:
            logger.info("The embedding classifier model is loaded.")
            return
        
        logger.info("Loading embedding classifier model.")
        self.load_embedding_model()
        self.embedding_classifier = EmbeddingClassifier(self.embedding_model)
        logger.info("Loading embedding classifier model is done.")

    def load_ensemble_model(self) -> None:
        if self.ensemble_model is not None:
            logger.info("The ensemble model is loaded.")
            return

        logger.info("Loading ensemble model.")
        self.load_embedding_classifier()
        self.load_tfidf_classifier()
        self.load_brands_classifier()
        self.load_translation_model()
        self.ensemble_model = EnsembleModel(
            self.brands_classifier,
            self.embedding_classifier,
            self.tfidf_classifier, 
            self.translation_model
        )
        logger.info("Loading ensemble model is done.")

    def test_pipeline(self) -> None:
        if self.df_test is None:
            logger.info("You need to give the path of the test dataset.")
            return

        logger.info("Start to test the whole pipeline on the test dataset")
        self.df_test["product_name"] = self.df_test["product_name"].astype(str)
        invoices = self.df_test["product_name"].tolist()

        self.load_ensemble_model()
        results = self.ensemble_model.run_ensemble(invoices)

        self.df_test["pred_segment"] = results["voted_segments"]
        self.df_test["pred_family"] = results["voted_families"]
        self.df_test["pred_class"] = results["voted_classes"]
        self.df_test["brand_segment"] = results["brand_tfidf_sim_preds"][0]
        self.df_test["brand_family"] = results["brand_tfidf_sim_preds"][1]
        self.df_test["brand_class"] = results["brand_tfidf_sim_preds"][2]
        self.df_test["clf_segment"] = results["tfidf_clf_preds"][0]
        self.df_test["clf_family"] = results["tfidf_clf_preds"][1]
        self.df_test["clf_class"] = results["tfidf_clf_preds"][2]
        self.df_test["embed_segment"] = results["embed_clf_preds"][0]
        self.df_test["embed_family"] = results["embed_clf_preds"][1]
        self.df_test["embed_class"] = results["embed_clf_preds"][2]
        self.df_test["confidence_rate"] = results["confidences"]
        self.df_test["confidence_level"] = get_confidence_level(results["confidences"])
        self.df_test.to_csv(ENSEMBLE_MODEL_OUTPUT_DATASET_PATH, index=False)

        true_segment = self.df_test["segment"].tolist()
        true_family = self.df_test["pred_family"].tolist()
        true_class = self.df_test["pred_class"].tolist()

        logger.info(f"Level segment: {accuracy_score(true_segment, results["voted_segments"])}")
        logger.info(f"Level family: {accuracy_score(true_family, results["voted_families"])}")
        logger.info(f"Level class: {accuracy_score(true_class, results["voted_classes"])}")  

    @override
    def run_inference(self, invoice_items: Union[str, List[str]]) -> Dict[str, Any]:
        self.load_ensemble_model()
        preds = self.ensemble_model.run_ensemble(invoice_items)
    
        df_preds = pd.DataFrame({
            "product_name": invoice_items,
            "pred_segment": preds["voted_segments"],
            "pred_family": preds["voted_families"],
            "pred_class": preds["voted_classes"],
            "brand_segment": preds["brand_tfidf_sim_preds"][0],
            "brand_family": preds["brand_tfidf_sim_preds"][1],
            "brand_class": preds["brand_tfidf_sim_preds"][2],
            "embed_segment": preds["embed_clf_preds"][0],
            "embed_family": preds["embed_clf_preds"][1],
            "embed_class": preds["embed_clf_preds"][2],
            "clf_segment": preds["tfidf_clf_preds"][0],
            "clf_family": preds["tfidf_clf_preds"][1],
            "clf_class": preds["tfidf_clf_preds"][2],
            "confidence_rate": preds["confidences"],
            "confidence_level": get_confidence_level(preds["confidences"])
        })
        df_preds.to_csv(ENSEMBLE_PIPELINE_OUTPUT_PATH, index=False)

        return preds

    @override
    def run_pipeline(self) -> None:
        if self.df_train is None:
            logger.warning("You need to pass `df_train_path`.")
            return

        logger.info("Starting to train `TF-IDF Classifier`.")
        self.load_tfidf_classifier()
        train_tfidf_model(self.tfidf_classifier, self.df_train)
        logger.info("`TF-IDF Classifier` training is done.")
        if self.df_test is not None:
            logger.info("`TF-IDF Classifier` accuracy on testing data.")
            test_tfidf_model(self.tfidf_classifier, self.df_test)

        logger.info("Starting to train `Brands Classifier`.")
        self.load_brands_classifier()
        train_tfidf_model(self.brands_classifier, self.df_train)
        logger.info("`Brands Classifier` training is done.")
        if self.df_test is not None:
            logger.info("`Brands Classifier` accuracy on testing data.")
            test_tfidf_model(self.brands_classifier, self.df_test)

        self.test_pipeline()
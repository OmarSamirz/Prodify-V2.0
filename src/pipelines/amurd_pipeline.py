import pandas as pd

from typing_extensions import override
from typing import Optional, List, Union, Dict, Any

from modules.logger import logger
from pipelines.pipeline import Pipeline
from queries import (
    create_table_query,
    classification_query, 
    generate_result_query, 
    scores_query,
    update_table_columns_query,
    merge_table_columns_query,
    delete_rows_not_in_source_query
)


class AmurdPipeline(Pipeline):

    def __init__(
        self,
        df_train_path: str,
        df_val_path: Optional[str] = None,
        df_test_path: Optional[str] = None,
        embedding_model_config_path: Optional[str] = None,
        translation_model_config_path: Optional[str] = None,
        tfidf_classifier_config_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            df_train_path, 
            df_val_path, 
            df_test_path,
            embedding_model_config_path,
            translation_model_config_path,
            tfidf_classifier_config_path
        )

    @override
    def load_dataframe(self, df_path: str) -> pd.DataFrame:
        return super().load_dataframe(df_path)

    @override
    def rename_columns(self, dataframe: pd.DataFrame, renamed_columns: Dict[str, str]) -> pd.DataFrame:
        return super().rename_columns(dataframe, renamed_columns)

    @override
    def apply_unicode_cleaning(self, dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
        return super().apply_unicode_cleaning(dataframe, col)

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
    def database_insertion(self, dataframe: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
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
    def load_translation_model(self) -> None:
        super().load_translation_model()

    @override
    def load_tfidf_model(self) -> None:
        super().load_tfidf_model()

    @override
    def translate_data(self, table_name: str, translated_col: str, new_col_name: str) -> None:
        super().translate_data(table_name, translated_col, new_col_name)

    @override
    def create_embeddings(self, table_name: str, embedding_col: str,new_table_name: str,embeddings_name: str = "embed_") -> None:
        super().create_embeddings(table_name, embedding_col, new_table_name, embeddings_name)

    def delete_rows_not_in_source(
        self,
        target_table_name: str,
        source_table_name: str,
        source_col: str,
        target_col: str, 
    ) -> None:
        logger.info(f"Deleting rows from table `{target_table_name}` using table `{source_table_name}`.")
        target_table_name = self.combine_db_table_names(target_table_name)
        source_table_name = self.combine_db_table_names(source_table_name)
        self.execute_query(
            delete_rows_not_in_source_query(
                target_table_name,
                source_table_name,
                source_col,
                target_col
            )
        )

    def merge_table_columns(
        self,
        target_table_name: str,
        source_table_name: str,
        target_cols: Union[str, List[str]],
        source_cols: Union[str, List[str]],
        target_join_col: str,
        source_join_col: str
    ) -> None:
        if isinstance(target_cols, List):
            target_cols = ", ".join(target_cols)
        if isinstance(source_cols, List):
            source_cols = ", ".join(source_cols)

        logger.info(f"Merging column(s) `{target_cols}` of table `{target_table_name}` with column(s) `{source_cols}` of table `{source_table_name}`.")
        target_table_name = self.combine_db_table_names(target_table_name)
        source_table_name = self.combine_db_table_names(source_table_name)
        queries = merge_table_columns_query(
            target_table_name,
            source_table_name,
            target_cols,
            source_cols,
            target_join_col,
            source_join_col
        )
        for query in queries.split(";"):
            query += ";"
            self.execute_query(query)

    def find_similarity(
        self,
        similarity_table_name: str,
        similarity_table_cols_name: List[str],
        similarity_table_cols_type: List[str],
        target_id: str,
        reference_id: str,
        embedding_table_1: str,
        embedding_table_2: str,
        embeddings_name: str = "embed_",
    ) -> None:
        if self.embedding_model is None:
            logger.info("Loading the embedding model")
            self.load_embedding_model()

        logger.info(f"Creating a new table `{similarity_table_name}` with columns `{similarity_table_cols_name}` on database `{self.td_db.database}`.")
        self.execute_query(create_table_query(similarity_table_name, similarity_table_cols_name, similarity_table_cols_type))

        embed_dim = self.embedding_model.model.get_sentence_embedding_dimension()
        vector_cols = ", ".join([f"{embeddings_name}{i}" for i in range(embed_dim)])
        vector_cols_quoted = ", ".join([f"'{embeddings_name}{i}'" for i in range(embed_dim)])

        logger.info(f"Add classification to table `{similarity_table_name}`")
        self.execute_query(
            classification_query(
                similarity_table_name,
                vector_cols,
                vector_cols_quoted,
                target_id,
                reference_id,
                embedding_table_1,
                embedding_table_2
            )
        )
        df_similarity = self.get_table(similarity_table_name)
        logger.info(f"A sample from `{similarity_table_name}`:\n{df_similarity.head()}")

    def get_model_scores(
        self,
        scores_table_name: str,
        result_table_name: str,
        output_table_name: str,
        observation_col: str,
        prediction_col: str,
        table_to_count: str,
        col_to_count: str
    ) -> None:
        scores_table_name = self.combine_db_table_names(scores_table_name)
        logger.info(f"Create a new table `{result_table_name}`.")
        result_table_name = self.combine_db_table_names(result_table_name)
        self.execute_query(generate_result_query(result_table_name, scores_table_name))
        
        num_labels = self.count_table_rows(table_to_count, col_to_count) + 1
        self.execute_query(
            scores_query(
                result_table_name.split(".")[1],
                output_table_name,
                observation_col,
                prediction_col,
                num_labels
            )
        )
        df_scores = self.get_table(output_table_name)
        df_scores = pd.DataFrame(df_scores)
        logger.info(f"The scores of the model:\n{df_scores}")

    def evaluate_embedding_model(
        self,
        similarity_score_table: str,
        similarity_score_cols_name: List[str],
        similarity_score_cols_types: List[str],
        target_id: str,
        reference_id: str,
        embedding_table_1: str,
        embedding_table_2: str,
        result_table_name: str,
        classification_metric_table: str,
        observation_col: str,
        prediction_col: str,
        table_to_count: str,
        col_to_count: str,
        embeddings_name: str = "embed_"
    ) -> None:
        logger.info("Starting to evaluate embedding model.")
        self.find_similarity(
            similarity_score_table,
            similarity_score_cols_name,
            similarity_score_cols_types,
            target_id,
            reference_id,
            embedding_table_1,
            embedding_table_2,
            embeddings_name
        )

        self.get_model_scores(
            similarity_score_table,
            result_table_name,
            classification_metric_table,
            observation_col,
            prediction_col,
            table_to_count,
            col_to_count
        )
        logger.info("Evaluating embedding model is completed.")
    
    def evaluate_tfidf_model(
        self,
        train_table_name: str,
        train_labels_table_name: str,
        test_table_name: str,
        test_labels_table_name: str,
        classes_table_name: str
    ) -> None:
        if self.df_test is None:
            raise ValueError("You need to set a value to `df_test` to use this function.")
        
        logger.info("Evaluating tfidf classifier model.")
        self.load_tfidf_model()

        self.merge_table_columns(
            train_labels_table_name,
            classes_table_name,
            ["product_id", "class_name"],
            "id",
            "class_name",
            "class_name"
        )
        self.merge_table_columns(
            test_labels_table_name,
            classes_table_name,
            ["product_id", "class_name"],
            "id",
            "class_name",
            "class_name"
        )

        df_train = self.get_table(train_table_name)
        df_train_labels = self.get_table(train_labels_table_name)
        X_train = df_train["translated_product_name"].tolist()
        y_train = df_train_labels["id"].tolist()

        logger.info("Training the model.")
        self.tfidf_classifier.fit(X_train, y_train)

        df_test = self.get_table(test_table_name)
        df_test_labels = self.get_table(test_labels_table_name)
        X_test = df_test["translated_product_name"].tolist()
        y_test = df_test_labels["id"].tolist()

        logger.info("Testing the model.")
        y_pred = self.tfidf_classifier.predict(X_test)

        df_pred = pd.DataFrame({
            "actual_class": y_test,
            "predicted_class": y_pred
        })
        self.database_insertion(df_pred, "tfidf_predictions")

        num_labels = self.count_table_rows(classes_table_name, "id")
        self.execute_query(
            scores_query(
                "tfidf_predictions",
                "tfidf_scores",
                "actual_class",
                "predicted_class",
                num_labels
            )
        )
        df_scores = self.get_table("tfidf_scores")
        logger.info(f"The scores of tfidf classifer model: {df_scores}")

        logger.info("Saving the model.")
        self.tfidf_classifier.save()

    def update_table_columns(
        self,
        target_table: str,
        source_table: str,
        target_col: str,
        source_col: str,
        target_condition_col: Union[str, List[str]],
        source_condition_col: Union[str, List[str]]
    ) -> None:
        logger.info(f"Combining column `{target_col}` of table `{target_table}` with column `{source_col}` of table `{source_table}`.")
        target_table = self.combine_db_table_names(target_table)
        source_table = self.combine_db_table_names(source_table)
        self.execute_query(
            update_table_columns_query(
                target_table,
                source_table,
                target_col,
                source_col,
                target_condition_col,
                source_condition_col
            )
        )

    def process_dataframe(self, dataframe: pd.DataFrame, df_type: str) -> pd.DataFrame:
        logger.info(f"Starting to process `{df_type}` dataframe")

        dataframe = self.rename_columns(dataframe, {"Item_Name": "product_name", "class": "class_name"})
        dataframe = self.apply_unicode_cleaning(dataframe, "product_name")
        dataframe["df_type"] = [df_type] * len(dataframe)
        dataframe["id"] = dataframe.index
        
        products_table = f"products_{df_type}"
        self.database_insertion(dataframe[["id", "product_name", "class_name", "df_type"]], products_table)
        self.cleanse_table_column(products_table, "product_name")

        self.dropna(products_table, ["product_name", "class_name"])
        self.drop_duplicates(products_table, "product_name")
        self.update_id(products_table, "id", "product_name")

        classes_table = f"classes_{df_type}"
        actual_classes_table = f"actual_classes_{df_type}"
        self.replicate_table(products_table, actual_classes_table, ["id AS product_id", "product_name", "class_name", "df_type"])
        self.create_table_class(classes_table, actual_classes_table, "id", "class_name", "class_name")
        self.drop_column(products_table, "class_name")

        logger.info(f"Finished processing `{df_type}` dataframe")

    @override
    def run_pipeline(self) -> None:
        logger.info("Running the Amurd Pipeline.")
        self.process_dataframe(self.df_train, "train")
        if self.df_val is not None:
            self.process_dataframe(self.df_val, "val")
            self.delete_rows_not_in_source("classes_val", "classes_train", "class_name", "class_name")
            self.update_table_columns("classes_val", "classes_train", "id", "id", "class_name", "class_name")
            self.delete_rows_not_in_source("actual_classes_val", "actual_classes_train", "class_name", "class_name")
            self.delete_rows_not_in_source("products_val", "actual_classes_val", "product_id", "id")
            self.update_id("products_val", "id", "product_name")
        if self.df_test is not None:
            self.process_dataframe(self.df_test, "test")
            self.delete_rows_not_in_source("classes_test", "classes_train", "class_name", "class_name")
            self.update_table_columns("classes_test", "classes_train", "id", "id", "class_name", "class_name")
            self.delete_rows_not_in_source("actual_classes_test", "actual_classes_train", "class_name", "class_name")
            self.delete_rows_not_in_source("products_test", "actual_classes_test", "product_id", "id")
            self.update_id("products_test", "id", "product_name")

        self.concatenate_tables("products", "*")
        self.create_table_from("classes", "classes_train", "*")

        self.update_id("products", "id", "product_name", ["product_name", "df_type"])
        self.update_table_columns("products_train", "products", "id", "id", ["product_name", "df_type"], ["product_name", "df_type"])

        self.concatenate_tables("actual_classes", "*")
        self.update_table_columns("actual_classes", "products_train", "product_id", "id", ["product_name", "df_type"], ["product_name", "df_type"])
        self.update_table_columns("actual_classes", "products_test", "product_id", "id", ["product_name", "df_type"], ["product_name", "df_type"])
        self.update_table_columns("actual_classes_train", "actual_classes", "product_id", "product_id", ["product_name", "df_type"], ["product_name", "df_type"])

        self.translate_data("products", "product_name", "translated_product_name")
        self.merge_table_columns("products_train", "products", "*", "translated_product_name", "id", "id")

        if self.df_val is not None:
            self.update_table_columns("products_val", "products", "id", "id", ["product_name", "df_type"], ["product_name", "df_type"])
            self.update_table_columns("actual_classes_val", "actual_classes", "product_id", "product_id", ["product_name", "df_type"], ["product_name", "df_type"])
            self.merge_table_columns("products_val", "products", "*", "translated_product_name", "id", "id")
        if self.df_test is not None:
            self.update_table_columns("products_test", "products", "id", "id", ["product_name", "df_type"], ["product_name", "df_type"])
            self.update_table_columns("actual_classes_test", "actual_classes", "product_id", "product_id", ["product_name", "df_type"], ["product_name", "df_type"])
            self.merge_table_columns("products_test", "products", "*", "translated_product_name", "id", "id")

        self.create_embeddings("products", "translated_product_name", "product_embeddings")
        self.create_embeddings("classes", "class_name", "class_embeddings")

        self.evaluate_embedding_model(
            "similarity_score",
            ["product_id", "class_id", "cosine_distance"],
            ["BIGINT", "BIGINT", "FLOAT"],
            "product_id",
            "class_id",
            "product_embeddings",
            "class_embeddings",
            "results",
            "classification_metrics",
            "actual_class",
            "predicted_class",
            "classes",
            "class_name"
        )

        self.evaluate_tfidf_model("products_train", "actual_classes_train", "products_test", "actual_classes_test", "classes_test")

        logger.info("Running the Amurd Pipeline is done.")
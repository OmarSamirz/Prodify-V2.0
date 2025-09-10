import pandas as pd

from typing_extensions import override
from typing import Optional, Dict, List, Any, Union

from modules.logger import logger
from pipelines.pipeline import Pipeline
from queries import combine_columns_query, create_table_from_query, create_unique_table_from_query


class GpcPipeline(Pipeline):

    def __init__(
        self,
        df_gpc_path: str,
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
        self.df_gpc = self.load_dataframe(df_gpc_path)

    @override
    def load_dataframe(self, csv_path: str) -> pd.DataFrame:
        return super().load_dataframe(csv_path)

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
    def load_translation_model(self) -> None:
        super().load_translation_model()

    @override
    def load_tfidf_model(self):
        return super().load_tfidf_model()

    @override
    def translate_data(self, table_name: str, translated_col: str, new_col_name: str) -> None:
        super().translate_data(table_name, translated_col, new_col_name)

    @override
    def create_embeddings(
        self,
        table_name: str,
        embedding_col: str,
        new_table_name: str,
        embeddings_name: str = "embed_",
    ) -> None:
        super().create_embeddings(table_name, embedding_col, new_table_name, embeddings_name)

    def combine_columns(
        self,
        table_name: str, 
        cols: List[str], 
        new_col_name: str, 
        replace_null: str = "",
        separate_columns: str = ", ",
    ) -> None:
        logger.info(f"Combine columns `{', '.join(cols)}` in talbe `{table_name}` with new column name `{new_col_name}`.")
        
        table_name = self.combine_db_table_names(table_name)
        sql_parts = [f"COALESCE({cols[0]}, '{replace_null}')"]
        for col in cols[1:]:
            sql_parts.append(
                f"CASE WHEN {col} IS NOT NULL AND {col} <> '' "
                f"THEN '{separate_columns}' || {col} ELSE '' END"
            )
        cols_expr = " || ".join(sql_parts)

        queries = combine_columns_query(table_name, cols_expr, new_col_name)
        for query in queries.split(";"):
            if query.strip():
                self.execute_query(query + ";")
        logger.info(f"Columns combination is done.")
        logger.info(f"Columns combination is done.")

    def create_unique_table_from(
        self,
        target_table_name: str,
        source_table_name: str,
        min_col: str,
        target_cols: Union[str, List[str]],
        group_by_col: str,
    ) -> None:
        if isinstance(target_cols, List):
            target_cols = ", ".join(target_cols)

        logger.info(f"Create table `{target_table_name}` from table `{source_table_name}` with min column `{min_col}` and target column(s) `{target_cols}` and grouped by `{group_by_col}`.")
        target_table_name = self.combine_db_table_names(target_table_name)
        source_table_name = self.combine_db_table_names(source_table_name)
        self.execute_query(
            create_unique_table_from_query(
                target_table_name,
                source_table_name,
                min_col,
                target_cols,
                group_by_col
            )
        )
        logger.info("Table is created successfully.")

    def map_gpc_codes(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        unique_codes = df[col].unique()
        mapping = {code: i for i, code in enumerate(unique_codes)}
        df[col] = df[col].map(mapping)

        return df
    
    @override
    def process_dataframe(self, dataframe: pd.DataFrame, df_type: str) -> pd.DataFrame:
        logger.info(f"Starting to process `{df_type}` dataframe")

        dataframe = self.rename_columns(dataframe, {"Item_Name": "product_name", "class": "class_name"})
        dataframe = self.apply_unicode_cleaning(dataframe, "product_name")
        dataframe["df_type"] = [df_type] * len(dataframe)
        dataframe["id"] = dataframe.index

        products_table = f"gpc_products_{df_type}"
        self.database_insertion(dataframe[["id", "product_name", "class_name", "df_type"]], products_table)
        self.cleanse_table_column(products_table, "product_name")

        self.dropna(products_table, ["product_name", "class_name"])
        self.drop_duplicates(products_table, "product_name")
        self.update_id(products_table, "id", "product_name")

        logger.info(f"Finished processing `{df_type}` dataframe")

    @override
    def run_pipeline(self) -> None:
        logger.info("Running the Amurd Pipeline.")

        self.process_dataframe(self.df_train, "train")
        if self.df_val is not None:
            self.process_dataframe(self.df_val, "test")
        if self.df_test is not None:
            self.process_dataframe(self.df_test, "test")

        self.concatenate_tables("gpc_products", "*")
        self.update_id("gpc_products", "id", "product_name", ["product_name", "df_type"])
        self.update_table_columns("gpc_products_train", "gpc_products", "id", "id", ["product_name", "df_type"], ["product_name", "df_type"])

        if self.df_val is not None:
            self.update_table_columns("gpc_products_val", "gpc_products", "id", "id", ["product_name", "df_type"], ["product_name", "df_type"])
        if self.df_test is not None:
            self.update_table_columns("gpc_products_test", "gpc_products", "id", "id", ["product_name", "df_type"], ["product_name", "df_type"])

        self.df_gpc.drop_duplicates(subset=["BrickTitle"], inplace=True)
        self.df_gpc.dropna(subset=["BrickTitle"], inplace=True)
        self.df_gpc = self.map_gpc_codes(self.df_gpc, "SegmentCode")
        self.df_gpc = self.map_gpc_codes(self.df_gpc, "FamilyCode")
        self.df_gpc = self.map_gpc_codes(self.df_gpc, "ClassCode")
        self.df_gpc = self.map_gpc_codes(self.df_gpc, "BrickCode")
        self.df_gpc["id"] = self.df_gpc.index

        self.database_insertion(self.df_gpc, "gpc")
        self.drop_column(
            "gpc",
            [
                "AttributeTitle",
                "AttributeDefinition",
                "AttributeValueCode",
                "AttributeValueTitle",
                "AttributeValueDefinition",
                "AttributeCode",
            ],
        )
        # self.drop_duplicates("gpc", "BrickTitle")
        # self.dropna("gpc", "BrickTitle")
        self.update_id("gpc", "id", "id")

        self.combine_columns(
            "gpc",
            [
                "SegmentTitle",
                "SegmentDefinition",
                "FamilyTitle",
                "FamilyDefinition",
                "ClassTitle",
                "ClassDefinition",
                "BrickTitle",
                "BrickDefinition_Includes",
                "BrickDefinition_Excludes"
            ],
            "classes_definition"
        )
        self.drop_table("gpc_train")
        self.create_table_from(
            "gpc_train",
            "gpc",
            [
                "id",
                "classes_definition",
                "SegmentCode",
                "FamilyCode",
                "ClassCode",
                "BrickCode"
            ]
        )

        self.translate_data("gpc_products", "product_name", "translated_product_name")

        self.create_embeddings("gpc_products", "translated_product_name", "product_embeddings")
        self.create_embeddings("gpc_train", "classes_definition", "gpc_train_embeddings")

        self.create_unique_table_from("gpc_segment", "gpc", "id", "SegmentTitle", "SegmentTitle")
        self.create_embeddings("gpc_segment", "SegmentTitle", "gpc_segment_embeddings")
        self.drop_table("gpc_segment")

        self.create_unique_table_from("gpc_family", "gpc", "id", "FamilyTitle", "FamilyTitle")
        self.create_embeddings("gpc_family", "FamilyTitle", "gpc_family_embeddings")
        self.drop_table("gpc_family")

        self.create_unique_table_from("gpc_class", "gpc", "id", "ClassTitle", "ClassTitle")
        self.create_embeddings("gpc_class", "ClassTitle", "gpc_class_embeddings")
        self.drop_table("gpc_class")

        self.create_unique_table_from("gpc_brick", "gpc", "id", "BrickTitle", "BrickTitle")
        self.create_embeddings("gpc_brick", "BrickTitle", "gpc_brick_embeddings")
        self.drop_table("gpc_brick")
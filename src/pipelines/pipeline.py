import swifter
import pandas as pd
from teradataml import *

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union

from modules.logger import logger
from modules.db import TeradataDatabase
from queries import (
    cleansing_query,
    num_rows_query,
    drop_table_query,
    drop_column_query,
    drop_duplicates_query,
    dropna_query,
    update_id_query,
    replicate_table_query,
    create_class_table_query,
    create_table_from_query,
    renumber_ids_new_table_query,
    update_table_columns_query
)
from utils import (
    load_embedding_model,
    load_translation_model,
    load_tfidf_model,
    unicode_clean,
)


class Pipeline(ABC):

    def __init__(
        self,
        df_train_path: str,
        df_val_path: Optional[str] = None,
        df_test_path: Optional[str] = None,
        embedding_model_config_path: Optional[str] = None,
        translation_model_config_path: Optional[str] = None,
        tfidf_classifier_config_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.td_db = TeradataDatabase()
        self.td_db.connect()

        self.df_train = self.load_dataframe(df_train_path)
        self.df_val = None if df_val_path is None else self.load_dataframe(df_val_path)
        self.df_test = None if df_test_path is None else self.load_dataframe(df_test_path)

        self.embedding_model = None
        self.translation_model = None
        self.tfidf_classifier = None
        self.embedding_model_config_path = embedding_model_config_path
        self.translation_model_config_path = translation_model_config_path
        self.tfidf_classifier_config_path = tfidf_classifier_config_path

    def __del__(self) -> None:
        self.td_db.disconnect()

    @abstractmethod
    def load_dataframe(self, df_path: str) -> pd.DataFrame:
        logger.info(f"Loading a CSV file of path `{df_path}`.")
        if df_path.suffix == ".csv":
            df = pd.read_csv(df_path)
        elif df_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(df_path)
        else:
            raise ValueError(f"This type `{df_path.suffix}` is not supported.")

        return df

    @abstractmethod
    def rename_columns(self, dataframe: pd.DataFrame, renamed_columns: Dict[str, str]) -> pd.DataFrame:
        logger.info(f"Renaming columns:\n{renamed_columns}")

        return dataframe.rename(columns=renamed_columns)

    @abstractmethod
    def apply_unicode_cleaning(self, dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
        logger.info(f"Applying unicode cleaning on column: `{col}`.")
        dataframe[col] = dataframe[col].swifter.apply(unicode_clean)

        return dataframe

    @abstractmethod
    def combine_db_table_names(self, table_name: str) -> str:
        return self.td_db.database + "." + table_name

    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        return self.td_db.execute_query(query)
    
    @abstractmethod
    def drop_column(self, table_name: str, cols: Union[str, List[str]]) -> None:
        if isinstance(cols, str):
            cols = [cols]

        table_name = self.combine_db_table_names(table_name)
        for col in cols:
            logger.info(f"Dropping column `{col}` in talbe `{table_name}`.")
            self.execute_query(drop_column_query(table_name, col))
    
    @abstractmethod
    def dropna(self, table_name: str, cols: Union[str, List[str]]) -> None:
        if isinstance(cols, str):
            cols = [cols]
        
        table_name = self.combine_db_table_names(table_name)
        for col in cols:
            logger.info(f"Dropping na values in talbe `{table_name}` for column `{col}`.")
            queries = dropna_query(table_name, col)
            for query in queries.split(";"):
                query += ";"
                self.execute_query(query)

    @abstractmethod
    def drop_duplicates(self, table_name: str, cols: Union[str, List[str]], id_name: str = "id") -> None:
        if isinstance(cols, str):
            cols = [cols]

        table_name = self.combine_db_table_names(table_name)
        for col in cols:
            logger.info(f"Dropping duplicates in talbe `{table_name}` for column `{col}`.")
            queries = drop_duplicates_query(table_name, col, id_name)
            for query in queries.split(";"):
                query += ";"
                self.execute_query(query)

    @abstractmethod
    def update_id(self, table_name: str, id_name: str, order_by_col: str, selected_cols: Optional[Union[str, List[str]]] = None) -> None:
        if isinstance(selected_cols, List):
            selected_cols = ", ".join(selected_cols)
        
        logger.info(f"Updatting `{id_name}` in table `{table_name}`.")
        table_name = self.combine_db_table_names(table_name)
        try:
            self.execute_query(update_id_query(table_name, id_name, order_by_col))
        except:
            queries = renumber_ids_new_table_query(table_name, id_name, selected_cols, order_by_col)
            for query in queries.split(";"):
                query += ";"
                self.execute_query(query)

    @abstractmethod
    def replicate_table(self, current_table_name: str, new_table_name: str, cols: Union[str, List[str]]) -> None:
        if isinstance(cols, List):
            cols = ", ".join(cols)

        logger.info(f"Creatting new table `{new_table_name}` from table `{current_table_name}`.")
        new_table_name = self.combine_db_table_names(new_table_name)
        current_table_name = self.combine_db_table_names(current_table_name)
        self.execute_query(replicate_table_query(current_table_name, new_table_name, cols))

    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        logger.info(f"Droping table `{table_name}` on database `{self.td_db.database}`.")
        table_name = self.combine_db_table_names(table_name)
        self.execute_query(drop_table_query(table_name))
        logger.info("Droping table is done.")

    @abstractmethod
    def create_table_class(
        self, 
        new_table_name: str,
        current_table_name: str,
        id_col: str, 
        distinct_col_name: str,
        selected_cols: Union[str, List[str]]
    ) -> None:
        if isinstance(selected_cols, List):
            cols = ", ".join(cols)
        
        logger.info(f"Create table `{new_table_name}` using talbe `{current_table_name}` with column(s) `{selected_cols}`.")
        new_table_name = self.combine_db_table_names(new_table_name)
        current_table_name = self.combine_db_table_names(current_table_name)
        self.execute_query(
            create_class_table_query(
                new_table_name,
                current_table_name,
                id_col,
                distinct_col_name,
                selected_cols
            )
        )

    @abstractmethod
    def get_table(self, table_name: str) -> pd.DataFrame:
        tdf = DataFrame.from_table(table_name)
        df = tdf.to_pandas()

        return df

    @abstractmethod
    def count_table_rows(self, table_name: str, col: str) -> int:
        table_name = self.combine_db_table_names(table_name)
        tdf = self.execute_query(num_rows_query(table_name, col))

        return tdf[0][col]

    @abstractmethod
    def database_insertion(self,dataframe: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        logger.info(f"Inserting data in table `{table_name}` on database `{self.td_db.database}`.")
        copy_to_sql(dataframe, table_name, self.td_db.database, if_exists=if_exists)
        logger.info(f"Insertion completed.")

    @abstractmethod
    def cleanse_table_column(self, table_name: str, col_name: str) -> None:
        logger.info(f"Starting data cleansing for column `{col_name}` in table `{table_name}` on database `{self.td_db.database}`.")
        table_name = self.combine_db_table_names(table_name)
        self.execute_query(cleansing_query(table_name, col_name))

    @abstractmethod
    def concatenate_tables(
        self, 
        table_name: str, 
        selected_cols: Union[str, List[str]], 
    ) -> None:
        logger.info(f"Concatenating tables of name `{table_name}`.")
        if isinstance(selected_cols, List):
            selected_cols = ", ".join(selected_cols)
        
        table_name = self.combine_db_table_names(table_name)
        
        selection_quries = [f"SELECT {selected_cols} FROM {table_name}_train"]

        if self.df_val is not None:
            selection_quries.append(f"SELECT {selected_cols} FROM {table_name}_val")
        if self.df_test is not None:
            selection_quries.append(f"SELECT {selected_cols} FROM {table_name}_test")
        
        union_query = " UNION ALL ".join(selection_quries)
        query = f"CREATE TABLE {table_name} AS ({union_query}) WITH DATA;"
        self.execute_query(query)
        logger.info(f"Concatenating tables is done.")

    @abstractmethod
    def create_table_from(self, target_table_name: str, source_table_name: str, target_cols: Union[str, List[str]]) -> None:
        if isinstance(target_cols, List):
            target_cols = ", ".join(target_cols)
        
        logger.info(f"Create new table `{target_table_name}` from table `{source_table_name}` using column(s) `{target_cols}`.")
        target_table_name = self.combine_db_table_names(target_table_name)
        source_table_name = self.combine_db_table_names(source_table_name)
        self.execute_query(create_table_from_query(target_table_name, source_table_name, target_cols))
        logger.info(f"Table `{target_table_name}` is created successfully.")
    
    @abstractmethod
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

    @abstractmethod
    def load_embedding_model(self) -> None:
        if self.embedding_model is not None:
            logger.info("The embedding model is loaded.")
            return
        if self.embedding_model_config_path is None:
            raise ValueError(f"You need to set a value for `embedding_model_config_path` to use this function.")

        logger.info("Loading embedding model.")
        self.embedding_model = load_embedding_model(self.embedding_model_config_path)
        logger.info("Loading embeddings model is done.")

    @abstractmethod
    def load_translation_model(self) -> None:
        if self.translation_model is not None:
            logger.info("The translation model is loaded.")
            return 
        if self.translation_model_config_path is None:
            raise ValueError(f"You need to set a value for `translation_model_config_path` to use this function.")

        logger.info("Loading translation model.")
        self.translation_model = load_translation_model(self.translation_model_config_path)
        logger.info("Loading translation model is done.")

    @abstractmethod
    def load_tfidf_model(self) -> None:
        if self.tfidf_classifier is not None:
            logger.info("The tfidf classifier model is loaded.")
            return
        if self.tfidf_classifier_config_path is None:
            raise ValueError(f"You need to set a value for `tfidf_classifier_config_path` to use this function.")
        
        logger.info("Loading tfidf classifier model.")
        self.tfidf_classifier = load_tfidf_model(self.tfidf_classifier_config_path)
        logger.info("Loading tfidf classifier model is done.")

    @abstractmethod
    def translate_data(self, table_name: str, translated_col: str, new_col_name: str) -> None:
        self.load_translation_model()
        df = self.get_table(table_name)
        logger.info(f"Starting to translate column `{translated_col}` of table `{table_name}`.")
        df[new_col_name] = df[translated_col].swifter.apply(self.translation_model.translate)
        df[new_col_name] = df[new_col_name].str.lower()
        logger.info("Translation Completed.")
        logger.info(f"Sample from translated data:\n{df.head(3)}")
        self.database_insertion(df, table_name)

    @abstractmethod
    def create_embeddings(self, table_name: str, embedding_col: str, new_table_name: str, embeddings_name: str = "embed_") -> None:
        self.load_embedding_model()

        df = self.get_table(table_name)

        logger.info(f"Starting to convert column `{embedding_col}` of table `{table_name}` to embeddings.")
        queries = df[embedding_col].tolist()
        embeddings = self.embedding_model.get_embeddings(queries)
        embeddings = embeddings.tolist()
        logger.info("Converting to embeddings completed.")

        emb_cols = pd.DataFrame(embeddings, columns=[f'{embeddings_name}{i}' for i in range(len(embeddings[0]))])
        df_expanded = pd.concat([df[["id"]], emb_cols], axis=1)
        self.database_insertion(df_expanded, new_table_name)

    @abstractmethod
    def process_dataframe(self, dataframe: pd.DataFrame, df_type: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def run_pipeline(self) -> None:
        ...
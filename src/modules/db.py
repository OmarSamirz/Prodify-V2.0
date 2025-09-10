import teradataml as tdml
from dotenv import load_dotenv
from teradataml import execute_sql

import os
from typing import Any, List, Dict

from constants import ENV_PATH
from modules.logger import logger


class TeradataDatabase:

    def __init__(self):
        load_dotenv(ENV_PATH)

        self.host = os.getenv("TD_HOST")
        self.database = os.getenv("TD_NAME")
        self.user = os.getenv("TD_USER")
        self.password = os.getenv("TD_PASSWORD")
        self.port = os.getenv("TD_PORT", 1025)
        self.connection = None
        self._schema_cache = None
        logger.info("TeradataDatabase instance initialized")

    def connect(self) -> None:
        logger.log_operation("Database connection", "started", {"host": self.host, "database": self.database})
        try:
            self.connection = tdml.create_context(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            logger.log_operation("Database connection", "completed")
        except tdml.OperationalError as e:
            logger.exception(e, "database connection")
            raise

    def disconnect(self) -> None:
        if self.connection:
            logger.log_operation("Database disconnection", "started")
            tdml.remove_context()
            self.connection = None
            logger.log_operation("Database disconection", "completed")

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        if not self.connection:
            logger.error("Database connection is not established")
            raise Exception("Database connection is not established. Call connect() first.")
        
        try:
            logger.log_operation("Query execution", "started", {"query": query[:100] + "..." if len(query) > 100 else query})
            result = execute_sql(query)

            columns = [desc[0] for desc in result.description] if result.description else []

            rows = result.fetchall()
            results = [
                {columns[i]: value for i, value in enumerate(row)}
                for row in rows
            ]

            logger.log_operation("Query execution", "completed", {"rows_returned": len(results)})
            return results
        except Exception as e:
            logger.exception(e, "query execution")
            raise Exception(f"Query execution failed: {str(e)}")

    def _get_sample_data(self, table: str) -> str:
        query = f"SELECT * FROM {table} SAMPLE 3;"
        try:
            logger.debug(f"Getting sample data for table: {table}")
            result = execute_sql(query)
            if result.rowcount == 0:
                logger.debug(f"No sample data available for table: {table}")
                return "No data available"

            rows = result.fetchall()
            columns = [col[0] for col in result.description]

            result = [f"column: {', '.join(columns)}"]

            for i, row in enumerate(rows):
                row_values = [str(val) if val is not None else "NULL" for val in row]
                result.append(f"row{i+1}: {', '.join(row_values)}")

            return "\n".join(result)
        except tdml.OperationalError as e:
            logger.exception(e, f"retrieving sample data for {table}")
            return "Error retrieving sample data"
        
    def get_schema(self) -> str:
        if not self.connection:
            logger.error("Database connection is not established")
            raise Exception("Databse connection is not established. Call connect() first.")
        
        if self._schema_cache:
            logger.debug("Returning cached schema.")
            return self._schema_cache
        
        logger.log_operation("Schema retrieval", "started")
        query = """
        SELECT t.tablename, c.columnname, c.columntype
        FROM dbc.tablesv as t
        JOIN dbc.columnsv as c
        ON t.tablename = c.tablename AND t.databasename = c.databasename
        WHERE t.databasename = 'amurd'
        AND t.TableKind = 'T'
        ORDER BY t.tablename
        """

        td_type_map = {
            "CV": "String",
            "D": "Numeric",
            "CF": "Numeric",
            "I": "Integer",
            "F": "Float",
            "BO": "BLOB",
        }

        try:
            result = execute_sql(query)
            if result.rowcount == 0:
                logger.warning("No schema data returned from database")
                raise
            
            rows = result.fetchall()
            schema = {}
            for row in rows:
                table_name = row[0]
                column_name = row[1]
                data_type = row[2]
                if table_name not in schema:
                    schema[table_name] = []
                schema[table_name].append(f"{column_name} ({td_type_map[data_type.strip()]})")

            schema_parts = []
            for table, columns in schema.items():
                table_schema = f"\n\nTable: {table}\nColumn:\n - " + "\n  - ".join(columns)

                sample_data = self._get_sample_data(table)
                table_schema += f"\n\nSample data:\n{sample_data}" if sample_data else "\nSample data: No data available"

                schema_parts.append(table_name)
            
            self._schema_cache = "\n".join(schema_parts)
            logger.log_operation("Schema retrieval", "completed", {"tables_count": len(schema)})
            return self._schema_cache
        except tdml.OperationalError as e:
            logger.exception(e, "retrieving schema")
            raise

    def _format_td_type(self, type_code: str, length: int, total_digits: int, fractional_digits: int) -> str:
        type_code = type_code.strip()
        if type_code == "CV":
            return f"VARCHAR({length})"
        elif type_code == "CF":
            return f"CHAR({length})"
        elif type_code == "D":
            return "DATE"
        elif type_code == "I":
            return "INTEGER"
        elif type_code == "F":
            return f"DECIMAL({total_digits}, {fractional_digits})"
        elif type_code == "BO":
            return "BLOB"
        else:
            return type_code

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        if exc_type:
            logger.exception(exc_value, "database context manager")
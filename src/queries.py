
from typing import List, Union

def num_rows_query(table_name: str, col: str) -> str:
    r"""Query to get the total number of rows for specific column."""
    return f"SELECT COUNT({col}) FROM {table_name}"

def create_table_query(table_name: str, cols: List[str], types: List[str]) -> str:
    r"""Query to create a new table by giving table name, columns and columns type."""
    pairs = [f"{c} {t}" for c, t in zip(cols, types)]
    cols_types = ", ".join(pairs)
    return f"CREATE TABLE {table_name} ({cols_types});"

def drop_table_query(table_name: str) -> str:
    """Query to drop a table by giving table name."""
    return f"""DROP TABLE {table_name}"""

def cleansing_query(table_name: str, column_name: str) -> str:
    r"""Cleansing query that converts text to lowercase, removes punctuation, extra spaces, and digits."""
    return rf"""
    UPDATE {table_name}
    SET {column_name} = LOWER(
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE({column_name}, '[[:digit:]]+', ''), 
                        '[-_/\\|]', ' '
                    ),
                    '[[:punct:]]', ' '
                ),
                '\s+', ' '
            )
        )
    );
    """

def generate_result_query(result_table_name: str, scores_table_name: str) -> str:
    r"""Generate an SQL query to retrieve model evaluation results."""
    return f"""
    CREATE TABLE {result_table_name} AS (
        SELECT
            p.id AS product_id,
            p.translated_product_name AS product_name,
            c.class_name AS predicted_class,
            a.class_name AS actual_class,
            r.cosine_distance AS similarity_score
        FROM {scores_table_name} r
        JOIN products p
            ON r.product_id = p.id
        JOIN classes c
            ON r.class_id = c.id
        JOIN actual_classes a
            ON a.product_id = p.id
    ) WITH DATA;
    """

def classification_query(
    result_table: str, 
    vector_cols: str, 
    vector_cols_quoted: str, 
    target_id: str, 
    reference_id: str, 
    embedding_table_1: str, 
    embedding_table_2: str,
) -> str:
    r"""Generate an SQL query for nearest-neighbor classification using cosine similarity."""
    return f"""
    INSERT INTO {result_table}
    WITH RankedDistances AS (
        SELECT
            o.Target_ID AS {target_id},
            o.Reference_ID AS {reference_id},
            o.Distance,
            ROW_NUMBER() OVER (PARTITION BY o.Target_ID ORDER BY o.Distance ASC) as rn
        FROM TD_VectorDistance (
            ON (SELECT id, {vector_cols} FROM {embedding_table_1}) AS TargetTable
            ON (SELECT id, {vector_cols} FROM {embedding_table_2}) AS ReferenceTable DIMENSION
            USING
                TargetIDColumn('id')
                RefIDColumn('id')
                TargetFeatureColumns({vector_cols_quoted})
                RefFeatureColumns({vector_cols_quoted})
                DistanceMeasure('cosine')
        ) AS o
    )
    SELECT
        {target_id},
        {reference_id},
        Distance
    FROM RankedDistances
    WHERE rn = 1;
    """

def scores_query(
    table_name: str, 
    output_table_name: str, 
    observation_col: str, 
    prediction_col: str, 
    num_labels: str
) -> str:
    r"""Generate an SQL query to evaluate classification performance using Teradata's TD_ClassificationEvaluator."""
    return f"""
    SELECT * FROM TD_ClassificationEvaluator (
    ON {table_name} AS InputTable
    OUT PERMANENT TABLE OutputTable({output_table_name})
    USING
        ObservationColumn('{observation_col}')
        PredictionColumn('{prediction_col}')
        NumLabels({num_labels})
    ) AS dt;
    """

def drop_column_query(table_name: str, col: str) -> str:
    return f"""
    ALTER TABLE {table_name}
    DROP {col};
    """

def merge_table_columns_query(
    target_table_name: str,
    source_table_name: str,
    target_cols: str,
    source_cols: str,
    target_join_col: str,
    source_join_col: str,
) -> str:
    if "," in target_cols:
        target_cols = [item.strip() for item in target_cols.split(",")]
        target_cols = ", ".join(f"{target_table_name.split(".")[1]}.{item}" for item in target_cols)
    else:
        target_cols = target_table_name.split(".")[1] + "." + target_cols

    if "," in source_cols:
        source_cols = [item.strip() for item in source_cols.split(",")]
        source_cols = ", ".join(f"{source_table_name.split(".")[1]}.{item}" for item in source_cols)
    else:
        source_cols = source_table_name.split(".")[1] + "." + source_cols

    target_table_name_new = target_table_name + "_new"
    return f"""
    CREATE TABLE {target_table_name_new} AS (
        SELECT {target_cols}, {source_cols}
        FROM {target_table_name}
        LEFT JOIN {source_table_name}
        ON {target_table_name.split(".")[1]}.{target_join_col} = {source_table_name.split(".")[1]}.{source_join_col}
    ) WITH DATA;
    DROP TABLE {target_table_name};
    RENAME TABLE {target_table_name_new} TO {target_table_name.split(".")[1]};
    """

def drop_duplicates_query(table_name: str, col: str, id_col: str) -> str:
    r"""Query to drop repeated rows for a specific column."""
    new_table_name = table_name + "_new"
    return f"""
    CREATE TABLE {new_table_name} AS (
        SELECT *
        FROM {table_name}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY {col} ORDER BY {id_col}) = 1
    ) WITH DATA;
    DROP TABLE {table_name};
    RENAME TABLE {new_table_name} TO {table_name.split(".")[1]};
    """

def dropna_query(table_name: str, col: str) -> str:
    r"""Query to drop rows with null values in a specific column."""
    new_table_name = table_name + "_new"
    return f"""
    CREATE TABLE {new_table_name} AS (
        SELECT *
        FROM {table_name}
        WHERE {col} IS NOT NULL
    ) WITH DATA;
    DROP TABLE {table_name};
    RENAME TABLE {new_table_name} TO {table_name.split(".")[1]};
    """

def update_id_query(table_name: str, id_col: str, order_by_col: str) -> str:
    return f"""
    UPDATE {table_name} t
    SET {id_col} = (
        SELECT new_id
        FROM (
            SELECT ROW_NUMBER() OVER (ORDER BY {order_by_col}) AS new_id, {id_col} AS old_id
            FROM {table_name}
        ) x
        WHERE x.old_id = t.{id_col}
    );
    """

def renumber_ids_new_table_query(table_name: str, id_name: str, cols: str, order_by_col: str) -> str:
    new_table_name = table_name + "_new"
    return f"""
    CREATE TABLE {new_table_name} AS (
        SELECT ROW_NUMBER() OVER (ORDER BY {order_by_col}) AS {id_name}, {cols}
        FROM {table_name}
    ) WITH DATA;
    DROP TABLE {table_name};
    RENAME TABLE {new_table_name} TO {table_name.split(".")[1]};
    """

def replicate_table_query(current_table_name: str, new_table_name: str, cols: str) -> str:
    return f"""
    CREATE TABLE {new_table_name} AS (
        SELECT {cols}
        FROM {current_table_name}
    ) WITH DATA;
    """

def create_class_table_query(
    new_table_name: str,
    current_table_name: str,
    id_col: str,
    distinct_col_name: str,
    selected_cols: str,
) -> None:
    return  f"""
    CREATE TABLE {new_table_name} AS (
        SELECT 
            ROW_NUMBER() OVER (ORDER BY class_name) - 1 AS {id_col},
            {selected_cols}
        FROM (
            SELECT DISTINCT {distinct_col_name}
            FROM {current_table_name}
        ) AS distinct_classes
    ) WITH DATA;
    """

def update_table_columns_query(
    target_table: str,
    source_table: str,
    target_column: str,
    source_column: str,
    target_condition_columns: Union[str, List[str]],
    source_condition_columns: Union[str, List[str]]
) -> str:
    where_clause = f"WHERE {target_table.split(".")[1]}.{target_condition_columns} = {source_table.split(".")[1]}.{source_condition_columns}"
    if isinstance(target_condition_columns, List) and isinstance(source_condition_columns, List):
        where_clause = "WHERE " + " AND ".join(f"{target_table.split(".")[1]}.{target_condition_columns[i]} = {source_table.split(".")[1]}.{source_condition_columns[i]}" 
                                               for i in range(len(target_condition_columns)))
    
    return f"""
    UPDATE {target_table}
    FROM {source_table}
    SET {target_column} = {source_table.split(".")[1]}.{source_column}
    {where_clause};
    """

def combine_columns_query(table_name: str, cols: str, new_col_name: str) -> str:
    new_table_name = table_name + "_new"
    return f"""
    CREATE TABLE {new_table_name} AS (
        SELECT t.*, 
            TRIM(
                BOTH ' ' FROM
                {cols}
            ) AS {new_col_name}
        FROM {table_name} t
    ) WITH DATA;
    DROP TABLE {table_name};
    RENAME TABLE {new_table_name} TO {table_name.split(".")[1]};
    """

def create_table_from_query(target_table_name: str, source_table_name: str, target_cols: str) -> str:
    return f"""
    CREATE TABLE {target_table_name} AS (
        SELECT {target_cols}
        FROM {source_table_name}
    ) WITH DATA;
    """

def create_unique_table_from_query(
    target_table_name: str, 
    source_table_name: str, 
    min_col: str,
    target_cols: str,
    group_by_col: str,
) -> None:
    return f"""
    CREATE TABLE {target_table_name} AS (
        SELECT MIN({min_col}) AS {min_col}, {target_cols}
        FROM {source_table_name}
        GROUP BY {group_by_col}
    ) WITH DATA;
    """

def delete_rows_not_in_source_query(
    target_table_name: str,
    source_table_name: str,
    source_col: str,
    target_col: str,
) -> str:
    return f"""
    DELETE FROM {target_table_name}
    WHERE {target_col} NOT IN (
        SELECT {source_col} FROM {source_table_name}
    );
    """
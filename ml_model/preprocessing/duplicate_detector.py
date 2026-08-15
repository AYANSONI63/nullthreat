import hashlib
import pandas as pd 


def find_duplicate_columns(df: pd.DataFrame) -> dict:
    """
    Find identical columns using hashing.
    Returns:
        {
            duplicate_column: original_column
        }
    """

    column_hashes = {}

    for column in df.columns:
        column_hash = hashlib.md5(
            pd.util.hash_pandas_object(
                df[column],
                index=False
            ).values.tobytes()
        ).hexdigest()

        column_hashes[column] = column_hash

    
    hash_groups = {}

    for column, hash_value in column_hashes.items():

        hash_groups.setdefault(hash_value, []).append(column)

    duplicate_map = {}

    for columns in hash_groups.values():

        if len(columns) < 2:
            continue

        original = columns[0]


        for duplicate in columns[1:]:

            if df[original].equals(df[duplicate]):

                duplicate_map[duplicate] = original

    
    return duplicate_map


def remove_duplicate_columns(
        df: pd.DataFrame,
        duplicate_map: dict,
) -> pd.DataFrame:
    
    """
    Remove duplicate columns.
    """
    df = df.copy()


    return df.drop(columns=list(duplicate_map.keys()))

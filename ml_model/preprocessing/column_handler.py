DROP_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "Title",
    "TLD",
]


def drop_unused_columns(df):
    return df.drop(columns=DROP_COLUMNS).copy()
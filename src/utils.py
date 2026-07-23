def dataset_summary(df):
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\nClass Distribution:")
    print(df["type"].value_counts())
    
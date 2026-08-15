import pandas as pd 

def validate_dataframe(df: pd.DataFrame) -> None:
    
    """
    Basic dataset validation
    """

    if df.empty:
        raise ValueError("Dataset is empty.")
    
    if df.isnull().sum().sum() != 0:
        raise ValueError("Dataset contains missing value.")
    
    if "label" not in df.columns:
        raise ValueError("'label' column not found.")
    

def print_dataset_summary(df: pd.DataFrame) -> None:

    """
    Display dataset summary.
    """

    print("=" * 60)
    print(f"Shape : {df.shape}")
    print("=" * 60)

    print(df.info())

    print("\nFirst 5 Rows\n")
    print(df.head())

    print("\nTarget Distribution\n")
    print(df["label"].value_counts())
    
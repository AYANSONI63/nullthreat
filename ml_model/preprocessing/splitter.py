from sklearn.model_selection import train_test_split
import pandas as pd 


def split_dataset(
        df: pd.DataFrame,
        target_column: str = 'label',
        test_size: float = 0.2,
        random_state: int = 42
):
    """
    Split the dataset into train, validation and test sets using stratified sampling.
    """

    X = df.drop(columns=target_column)
    y = df[target_column]

    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )
    
    return (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
    )
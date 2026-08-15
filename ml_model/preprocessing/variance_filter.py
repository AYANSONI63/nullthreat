import pandas as pd 
from sklearn.feature_selection import VarianceThreshold


def compute_variance_filter(
    df: pd.DataFrame,
    threshold: float=0.0
):
    
    """
    Learn which features should be kept based on variance.
    """

    selector = VarianceThreshold(
        threshold=threshold
    )

    selector.fit(df)

    selected_features = df.columns[
        selector.get_support()
    ].tolist()

    removed_features = [
        column
        for column in df.columns
        if column not in selected_features
    ]


    return selected_features, removed_features



def apply_variance_filter(
    df: pd.DataFrame,
    selected_features: list  
) -> pd.DataFrame:
    
    return df[selected_features].copy()

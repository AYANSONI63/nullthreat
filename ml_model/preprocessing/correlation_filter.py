import json
import pandas as pd 
import numpy as np

from sklearn.feature_selection import mutual_info_classif



def compute_correlation_filter(
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.95
):
    """
    Finds highly correlated features and removes the weaker feature
    using Mutual Information.
    """

    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )


    mi_scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi_scores = pd.Series(
        mi_scores,
        index=X.columns
    )

    removed_features = set()

    decision_report = {}

    for column in upper_triangle.columns:

        correlated_features = upper_triangle.index[
            upper_triangle[column] > threshold
        ].tolist()

        for feature in correlated_features:

            if(
                feature in removed_features
                or column in removed_features
            ):
                continue

            feature_mi = mi_scores[feature]
            column_mi = mi_scores[column]

            if feature_mi >= column_mi:

                removed_features.add(column)

                decision_report[column] = {
                    "kept": feature,
                    "correlation": float(
                        upper_triangle.loc[
                            feature,
                            column
                        ]
                    ),
                    "removed_mi": float(column_mi),
                    "kept_mi": float(feature_mi)
                }
            
            else:

                removed_features.add(feature)

                decision_report[feature] = {
                    "kept": column,
                    "correlation": float(
                        upper_triangle.loc[
                            feature,
                            column
                        ]
                    ),
                    "removed_mi": float(feature_mi),
                    "kept_mi": float(column_mi)
                }

    selected_features = [
        column
        for column in X.columns
        if column not in removed_features
    ]

    return (
        selected_features,
        list(removed_features),
        decision_report
    )




def apply_correlation_filter(
        df: pd.DataFrame,
        selected_features: list
):
    return df[selected_features].copy()
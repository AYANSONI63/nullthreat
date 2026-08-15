import pandas as pd 

from sklearn.feature_selection import mutual_info_classif


def compute_mutual_information(
        X: pd.DataFrame,
        y: pd.Series
):
    """
    Compute Mutual Information score for every feature
    and return the ranking.
    """

    mi_scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi_scores = pd.Series(
        mi_scores,
        index=X.columns,
        name="mi_score" 
    )

    mi_scores = mi_scores.sort_values(
        ascending=False
    )

    ranking = []

    for rank, (feature, score) in enumerate(
        mi_scores.items(),
        start=1
    ):
        
        ranking.append(
            {
                "rank":rank,
                "feature": feature,
                "mi_score": float(score)
            }
        )

    return  mi_scores, ranking
import pandas as pd 
import numpy as np

from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel


def select_features(
        X: pd.DataFrame,
        y: pd.Series,
        threshold='median',
        random_state: int = 42
):
    
    """
    Select important features using XGBoost Feature Importance.
    """

    model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1
    )


    model.fit(X, y)


    selector = SelectFromModel(
        estimator=model,
        threshold=threshold,
        prefit=True
    )


    selected_columns = X.columns[selector.get_support()].tolist()


    removed_columns = [
        col
        for col in X.columns
        if col not in selected_columns
    ]

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)


    ranking = []


    for rank, (feature, score) in enumerate(
        importance.items(),
        start=1
    ):
        ranking.append(
            {
                "rank":rank,
                "feature":feature,
                "importance":float(score)
            }
        )
    
    threshold_value = float(
    np.median(model.feature_importances_)
    )
    

    return (
        model,
        selector,
        selected_columns,
        removed_columns,
        ranking,
        threshold_value
    )
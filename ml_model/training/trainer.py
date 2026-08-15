from xgboost import XGBClassifier


def train_xgboost(
    X_train,
    y_train,
    params=None
):
    
    """
    Train an XGBoost classifier
    """

    if params is None:

        params = {
            
            "n_estimators": 300,
            
            "max_depth": 6,
            
            "learning_rate": 0.1,
            
            "subsample": 0.8,
            
            "colsample_bytree": 0.8,
            
            "objective": "binary:logistic",
            
            "eval_metric": "logloss",
            
            "random_state": 42,
            
            "n_jobs": -1

        }

    model = XGBClassifier(**params)


    model.fit(
        X_train,
        y_train
    )


    return model 
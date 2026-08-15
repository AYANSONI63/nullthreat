import pandas as pd 

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(
    model,
    X,
    y      
):
    
    """
    Evaluate a trained binary classification model.

    Parameters
    ----------
    model : trained model
        Fitted classification model.

    X : pd.DataFrame
        Feature dataset.

    y : pd.Series
        True labels.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    # Prediction
    y_pred = model.predict(X)


    # Probability of positive class
    y_proba = model.predict_proba(X)[:,1]


    # Metrics
    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        y_proba
    )

    cm = confusion_matrix(
        y,
        y_pred
    )



    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }

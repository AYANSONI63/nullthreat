from typing import List, Dict


def build_feature_selection_artifact(
    ranking: List[Dict],
    selected_features: List[str],
    removed_features: List[str],
    threshold_type: str,
    threshold_value: float,
    random_state: int = 42
):
    """
    Build the feature selection artifact.
    """

    decision_report = {}

    for item in ranking:

        feature = item["feature"]
        importance = item["importance"]

        selected = feature in selected_features

        decision_report[feature] = {
            "importance": importance,
            "threshold": threshold_value,
            "decision": (
                "Selected"
                if selected
                else "Removed"
            ),
            "reason": (
                "Importance above threshold."
                if selected
                else "Importance below threshold."
            )
        }

    artifact = {

        "method": "XGBoost Feature Selection",

        "random_state": random_state,

        "threshold_type": threshold_type,

        "threshold_value": threshold_value,

        "total_input_features": len(ranking),

        "selected_feature_count": len(selected_features),

        "removed_feature_count": len(removed_features),

        "selected_features": selected_features,

        "removed_features": removed_features,

        "feature_importance": ranking,

        "decision_report": decision_report

    }

    return artifact
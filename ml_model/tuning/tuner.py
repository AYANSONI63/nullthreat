import json

import optuna

from pathlib import Path


from ml_model.training.trainer import train_xgboost
from ml_model.training.evaluator import evaluate_model


# Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# Objective Function 


def objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val
):
    
    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            600
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10 
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True
        ),


        # Fixed Parameters

        "objective": "binary:logistic",

        "eval_metric": "logloss",

        "random_state": 42,

        "n_jobs": -1
    }


    # Train model 

    model = train_xgboost(
        X_train,
        y_train,
        params=params
    )



    # Evaluate Validation set 

    metrics = evaluate_model(
        model,
        X_val,
        y_val
    )



    # We optimize F1 rather than accuracy.
    #
    # This prevents the tuner from focusing only on the
    # already extremely high accuracy.
    #
    # For malicious URL detection, balancing precision
    # and recall is more meaningful.
    
    return metrics["f1_score"]



def run_tuning(
    X_train, 
    y_train,
    X_val,
    y_val,
    n_trials=50
):
    
    print("=" * 60)
    print("XGBoost Hypermeter Tuning")
    print("=" * 60)


    print("\nTraining shape:")
    print(X_train.shape)


    print("\nValidation shape:")
    print(X_val.shape)


    print("\nNumber of trials:")
    print(n_trials)



    # Create Optuna Study 

    sampler = optuna.samplers.TPESampler(
    seed=42
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="nullthreat_xgboost_13_features",
        sampler=sampler
    )



    # Run Optimization 

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val
        ),
        n_trials=n_trials
    )



    # Best Result

    print("\n" + "=" * 60)
    print("Best Trial")
    print("=" * 60)

    print(
        f"\nBest F1 Score: "
        f"{study.best_value}"
    )

    print("\nBest Parameters:")


    for parameters, value in study.best_params.items():

        print(
            f"{parameters}: {value}"
        )

    

    # Save all trial results
    trials_path = ARTIFACTS_DIR / "xgboost_trials.csv"

    study.trials_dataframe().to_csv(
        trials_path,
        index=False
    )

    
    # Save Best Parameters 


    output_path = (
        ARTIFACTS_DIR / "xgboost_best_params.json"
    )


    with open(
        output_path,
        "w"
    ) as file:
        
        json.dump(
            study.best_params,
            file,
            indent=4
        )

    

    print("\nBest parameters saved to:")
    print(output_path)


    print("=" * 60)


    return study 
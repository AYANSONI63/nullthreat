import pandas as pd 
from pathlib import Path 
import joblib
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier



BASE_DIR = Path(__file__).parent 
ARTIFACT_DIR = BASE_DIR / "artifacts" 


def load_data():
    
    X_train = pd.read_csv(ARTIFACT_DIR / "X_train_selected.csv")
    y_train = pd.read_csv(ARTIFACT_DIR / "y_train.csv")["label"]
    X_test = pd.read_csv(ARTIFACT_DIR / "X_test_selected.csv")
    y_test = pd.read_csv(ARTIFACT_DIR / "y_test.csv")["label"]

    return X_train,X_test,y_train,y_test


def train_model(X_train,y_train):
    
    model = XGBClassifier(random_state=42)
    model.fit(X_train,y_train)

    return model


def evaluate_model(model, X_test, y_test):

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, pred))



def main():

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, ARTIFACT_DIR / "xgb_model.pkl")



if __name__ == "__main__":
    main()
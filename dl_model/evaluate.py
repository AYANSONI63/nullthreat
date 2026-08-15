import json 
import joblib
import numpy as np 
import torch 
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.dataset import URLDataset
from dl_model.model import URLClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "dl_model" / "artifacts"


encoder = joblib.load(
    ARTIFACTS / "label_encoder.pkl"
)

with open(ARTIFACTS / "char2idx.json", "r") as f:
    char2idx = json.load(f)



X_test = np.load(ARTIFACTS / "X_test.npy")
y_test = np.load(ARTIFACTS / "y_test.npy")


test_dataset = URLDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False
)

# Defining the hypermeters 

VOCAB_SIZE = len(char2idx)

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3


# Creating the model 

model = URLClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)


# Loading the BEST MODEL 

BEST_MODEL_PATH = ARTIFACTS / "bilstm_best_model.pth"

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()



all_predictions = []
all_labels = []
criterion = nn.CrossEntropyLoss()

test_running_loss = 0.0 

with torch.no_grad():


    for batch_urls, batch_labels in test_loader:

        batch_urls = batch_urls.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_urls)

        loss = criterion(outputs, batch_labels)

        test_running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)


        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            batch_labels.cpu().numpy()
        )


test_loss = test_running_loss / len(test_loader)


# Test Accuracy 

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

test_recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)


report = classification_report(
    all_labels,
    all_predictions,
    target_names=encoder.classes_
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("=" * 60)
print("Test Results")
print("=" * 60)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_accuracy:.4f}")
print(f"Test Precision : {test_precision:.4f}")
print(f"Test Recall    : {test_recall:.4f}")
print(f"Test F1 Score  : {test_f1:.4f}")

print("\nClassification Report")
print(report)

print("\nConfusion Matrix")
print(cm)
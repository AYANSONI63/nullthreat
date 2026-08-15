import numpy as np 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from src.dataset import URLDataset
from dl_model.model import URLClassifier
from pathlib import Path
import joblib
import json 
from tqdm import tqdm

# Load Training Data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "dl_model" / "artifacts"
encoder = joblib.load(PROJECT_ROOT / "dl_model" / "artifacts" / "label_encoder.pkl")

# Loading char2idx.json 
with open(PROJECT_ROOT / "dl_model" / "artifacts" / "char2idx.json", "r") as f:
    char2idx = json.load(f)

X_train = np.load(ARTIFACTS / "X_train.npy")
y_train = np.load(ARTIFACTS / "y_train.npy")

X_val = np.load(ARTIFACTS / "X_val.npy")
y_val = np.load(ARTIFACTS / "y_val.npy")


# print(X_train.shape)
# print(y_train.shape)

# print(X_val.shape)
# print(y_val.shape)

# print(X_train.dtype)
# print(y_train.dtype)

# # checking the labels 

# print(np.unique(y_train))
# print(encoder.classes_)


train_dataset = URLDataset(X_train, y_train)
val_dataset = URLDataset(X_val, y_val)

url, label = train_dataset[0]

print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
# print(url.shape)
# print(label)
# print(type(url))
# print(type(label))


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False
)

# Verifying...

# batch_urls, batch_labels = next(iter(train_loader))

# print(batch_urls.shape)
# print(batch_labels.shape)

# print(batch_urls.dtype)
# print(batch_labels.dtype)


# Hyperparameters 

VOCAB_SIZE = len(char2idx)
EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3

model = URLClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)


# Choose Device(CPU/GPU)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print(f"\nUsing device: {device}")

# Loss Function 

criterion = nn.CrossEntropyLoss()

# Update weights

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Scheduler 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# Best Validation loss seen so far 
best_val_loss = float('inf')

# Path to save best model 
BEST_MODEL_PATH =   ARTIFACTS / "bilstm_best_model.pth"

# Early Stopping
PATIENCE = 3
epochs_without_improvement = 0

# Storing training History 
train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

# Skeleton of the training loop 
EPOCHS = 10


# Model architecture 
print("\n",model)

print("=" * 60)
print("Starting Training...")
print("=" * 60)

for epoch in range(EPOCHS):

    running_loss = 0.0
    correct = 0
    total = 0

    model.train()

    for batch_urls, batch_labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"
        ):

        batch_urls = batch_urls.to(device)
        batch_labels = batch_labels.to(device)


        optimizer.zero_grad()

        outputs = model(batch_urls)

        loss = criterion(outputs, batch_labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        correct += (predicted == batch_labels).sum().item()

        total += batch_labels.size(0)


    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total


    # Validation 

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_urls, batch_labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"
            ):

            batch_urls = batch_urls.to(device)
            batch_labels = batch_labels.to(device)


            outputs = model(batch_urls)

            loss = criterion(outputs, batch_labels)


            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            val_correct += (predicted == batch_labels).sum().item() 

            val_total += batch_labels.size(0)

    
    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / val_total


    #Scheduler
    scheduler.step(val_loss)


    if val_loss < best_val_loss:
        
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        epochs_without_improvement = 0

        print(
            f"\n✓ Best model saved at Epoch {epoch + 1}"
            f"\nBest model saved! Validation Loss: {val_loss:.4f}")
    
    else:
        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )
    

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break

# Saving the training hoistory

history = {
    "train_loss": train_losses,
    "train_accuracy": train_accuracies,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
}

with open(ARTIFACTS / "training_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("\nTraining history saved successfully.")
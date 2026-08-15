import torch 
import json 
import joblib
from pathlib import Path

from dl_model.model import URLClassifier
from src.tokenizer import preprocess_single_url



PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = PROJECT_ROOT / "dl_model" / "artifacts"


# Load Artificats 

encoder = joblib.load(
    ARTIFACTS / "label_encoder.pkl"
)

with open(ARTIFACTS / "char2idx.json", "r") as f:

    char2idx = json.load(f)


# Hyperparameters

VOCAB_SIZE = len(char2idx)

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.3   


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Model Building 

model = URLClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

# Load Checkpoint

BEST_MODEL_PATH = ARTIFACTS / "bilstm_best_model.pth"

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()    


def predict(url):

    input_array = preprocess_single_url(url,char2idx, 256)

    input_tensor = torch.tensor(input_array, dtype=torch.long)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

        confidence = confidence.item() * 100


    predicted_class = predicted.item()
    
    predicted_label = encoder.inverse_transform(
        [predicted_class]
    )[0]


    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 2)
    }




if __name__ == "__main__":

    url = input("Enter URL: ")

    prediction = predict(url)

    print(f"\nPrediction: {prediction}")

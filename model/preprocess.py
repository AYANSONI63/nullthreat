import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os 
from imblearn.over_sampling import SMOTE


print("Loading dataset...")


if not os.path.exists("dataset/raw_phiusiil.csv"):
    print("ERROR: raw_phiusiil.csv not found in dataset folder")
    exit()


# Load PhiUSIIL dataset 

phiusiil_df = pd.read_csv("dataset/raw_phiusiil.csv")
print(f"PhiUSIIL loaded : {len(phiusiil_df)} rows")


# Cleaning PhiUIIL dataset

print("\nCleaning PhiUSIIL dataset")

phiusiil_df = phiusiil_df.drop(columns=['FILENAME'])

# Extracting Numerical features from the Domain column 

phiusiil_df['DomainHasNumbers'] = phiusiil_df['Domain'].str.contains(r'\d').astype(int)
phiusiil_df['DomainHyphenCount'] = phiusiil_df['Domain'].str.count('-')
phiusiil_df['DomainWordCount'] = phiusiil_df['Domain'].str.split('.').str.len()


# Dropping raw Domain column

phiusiil_df = phiusiil_df.drop(columns=['Domain'])

# Dropping missing values 

phiusiil_df = phiusiil_df.dropna()

phiusiil_df = phiusiil_df[phiusiil_df['label'].isin([0,1])] # Gives the only rows where the label is 0 or 1 


print(f"Phiusiil_df after cleaning: {len(phiusiil_df)} rows")
print(f"Label Distribution :\n{phiusiil_df['label'].value_counts()}")

# Fix typos in PhiUSIIL column names
phiusiil_df = phiusiil_df.rename(columns={
    'NoOfDegitsInURL': 'NoOfDigits',
    'DegitRatioInURL': 'DigitRatioInURL',
    'NoOfOtherSpecialCharsInURL': 'NoOfSpecialCharsInURL'
})



print("PhiUSIIL columns:", phiusiil_df.columns.tolist())


# Dropping string column

print("\nDropping string columns...")


string_cols_to_drop = []

for col in ['URL', 'Title', 'Robots', 'TLD']:
    if col in phiusiil_df.columns:
        string_cols_to_drop.append(col)


if string_cols_to_drop:
    phiusiil_df = phiusiil_df.drop(columns=string_cols_to_drop)
    print(f"Dropped: {string_cols_to_drop}")

print(f"Final columns: {phiusiil_df.shape[1]}")


print("\nSeprating features and labels...")

X = phiusiil_df.drop(columns=['label'])
y = phiusiil_df['label']


# Saving column name before everything converts to numpy

feature_cols = X.columns.tolist()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Total features: {len(feature_cols)}")



# Train test split 

print("\nSplitting into train and test...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")
print(f"Train label distribution:\n{y_train.value_counts()}")


# Auto-detecting binary cols from original X_train BEFORE scaling

binary_cols = []
for col in X_train.columns:
    unique_vals = set(X_train[col].dropna().unique())
    if unique_vals.issubset({0, 1, 0.0, 1.0}):
        binary_cols.append(col)


# StandardScaler Pipeline 

print("\nBuilding StandardScaler Pipeline...")


numerical_cols = X_train.columns.tolist()

preprocessor = ColumnTransformer(transformers=[('scaler', StandardScaler(), numerical_cols)], remainder='drop')

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fitting on train only 

X_train_scaled = pipeline.fit_transform(X_train)
X_test_scaled = pipeline.transform(X_test)

print(f"Scaled train shape: {X_train_scaled.shape}")
print(f"Scaled test shape: {X_test_scaled.shape}")


# SMOTE on scaled traning data only 


print("\nApplying SMOTE on traning data...")

smote = SMOTE(k_neighbors=5, random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE: {X_train_scaled.shape[0]} rows")
print(f"After SMOTE: {X_train_balanced.shape[0]} rows")
print(f"Balanced Label distribution:\n{pd.Series(y_train_balanced).value_counts()}")



# Round them back after SMOTE

X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=feature_cols)
X_train_balanced_df[binary_cols] = X_train_balanced_df[binary_cols].round()
X_train_balanced = X_train_balanced_df.values


print("\nSaving file...")


os.makedirs("model/saved_model", exist_ok=True)


# Saving processed array for TensorFlow training 

np.save("dataset/X_train.npy", X_train_balanced)
np.save("dataset/X_test.npy", X_test_scaled)
np.save("dataset/y_train.npy", y_train_balanced)
np.save("dataset/y_test.npy", y_test.values)

# Saving pipeline for FastAPI - load and preprocesses new url

with open("model/saved_model/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)


# Saving feature column name for FastAPI to align inputs 

with open("model/saved_model/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)


print("Saved: dataset/X_train.npy")
print("Saved: dataset/X_test.npy")
print("Saved: dataset/y_train.npy")
print("Saved: dataset/y_test.npy")
print("Saved: model/saved_model/pipeline.pkl")
print("Saved: model/saved_model/feature_cols.pkl")

print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print(f"Total features: {X_train_balanced.shape[1]}")
print(f"Training samples: {X_train_balanced.shape[0]}")
print(f"Testing samples: {X_test_scaled.shape[0]}")
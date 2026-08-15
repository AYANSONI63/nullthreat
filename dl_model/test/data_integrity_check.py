from pathlib import Path
import pandas as pd

# ----------------------------
# Load the original CSV
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "malicious_phish.csv"   # Change if your CSV path is different

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Original Dataset")
print("=" * 60)

print(f"Total Samples : {len(df)}")
print(f"Duplicate URLs: {df.duplicated(subset='url').sum()}")

print("\nClass Distribution:")
print(df["type"].value_counts())

# -------------------------------------------------
# Remove duplicates exactly like preprocess.py
# -------------------------------------------------

df = df.drop_duplicates(subset="url")

print("\nAfter Removing Duplicates")
print(f"Remaining Samples : {len(df)}")

# -------------------------------------------------
# Train / Validation / Test Split
# -------------------------------------------------

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["type"],
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["type"],
    random_state=42,
    shuffle=True
)

print("\nDataset Sizes")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

# -------------------------------------------------
# Duplicate URL Check
# -------------------------------------------------

train_urls = set(train_df["url"])
val_urls = set(val_df["url"])
test_urls = set(test_df["url"])

train_val_overlap = train_urls & val_urls
train_test_overlap = train_urls & test_urls
val_test_overlap = val_urls & test_urls

print("\n" + "=" * 60)
print("URL Overlap Check")
print("=" * 60)

print(f"Train <-> Validation : {len(train_val_overlap)}")
print(f"Train <-> Test       : {len(train_test_overlap)}")
print(f"Validation <-> Test  : {len(val_test_overlap)}")

assert len(train_val_overlap) == 0, "Duplicate URLs between Train and Validation!"
assert len(train_test_overlap) == 0, "Duplicate URLs between Train and Test!"
assert len(val_test_overlap) == 0, "Duplicate URLs between Validation and Test!"

print("✓ No duplicate URLs across splits.")

# -------------------------------------------------
# Label Distribution
# -------------------------------------------------

print("\n" + "=" * 60)
print("Label Distribution")
print("=" * 60)

print("\nTrain")
print(train_df["type"].value_counts(normalize=True).sort_index())

print("\nValidation")
print(val_df["type"].value_counts(normalize=True).sort_index())

print("\nTest")
print(test_df["type"].value_counts(normalize=True).sort_index())

# -------------------------------------------------
# Missing Values
# -------------------------------------------------

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())

assert df["url"].isnull().sum() == 0
assert df["type"].isnull().sum() == 0

print("✓ No missing URLs or labels.")

# -------------------------------------------------
# Empty URLs
# -------------------------------------------------

empty_urls = (df["url"].str.strip() == "").sum()

print(f"\nEmpty URLs : {empty_urls}")

assert empty_urls == 0

print("✓ No empty URLs.")

print("\n" + "=" * 60)
print("ALL DATA INTEGRITY CHECKS PASSED")
print("=" * 60)
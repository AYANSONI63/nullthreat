import pandas as pd 



df = pd.read_csv("dataset/raw_phiusiil.csv")

print("=" * 50)
print("BASIC INFO")
print("=" * 50)
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\n" + "=" * 50)
print("ALL COLUMN NAMES")
print("=" * 50)
for i, col in enumerate(df.columns.tolist()):
    print(f"{i+1}. {col}")

print("\n" + "=" * 50)
print("LABEL DISTRIBUTION")
print("=" * 50)
print(df['label'].value_counts())
print("\n1 = Legitimate")
print("0 = Phishing")

print("\n" + "=" * 50)
print("MISSING VALUES PER COLUMN")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("FIRST 3 ROWS")
print("=" * 50)
print(df.head(3))

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())
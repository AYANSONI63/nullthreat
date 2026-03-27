import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import pickle
import os 
from utils import is_ip_domain


print("Loading dataset...")


if not os.path.exists("dataset/raw_phiusiil.csv"):
    print("ERROR: raw_phiusiil.csv not found in dataset folder")
    exit()

if not os.path.exists("dataset/raw_urlhaus.csv"):
    print("ERROR: raw_urlhaus.csv not found in dataset folder")
    exit()


# Load PhiUSIIL dataset 

phiusiil_df = pd.read_csv("dataset/raw_phiusiil.csv")
print(f"PhiUSIIL loaded : {len(phiusiil_df)} rows")

# Reading raw line first to find the header

with open("dataset/raw_urlhaus.csv", "r") as f:
    lines = f.readlines()

# finding the header line starts with # id 

header_line = None 
for line in lines:
    if line.startswith("# id"):
        header_line = line.strip().lstrip('# ')
        break

# Parse column names
columns = [col.strip() for col in header_line.split(',')]

urlhaus_df = pd.read_csv("dataset/raw_urlhaus.csv", comment='#', names=columns, on_bad_lines='skip')
print(f"URLhaus loaded : {len(urlhaus_df)} rows")


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
    'DomainHasHyphen': 'DomainHyphenCount',
    'NoOfOtherSpecialCharsInURL': 'NoOfSpecialCharsInURL'
})


# Cleaning URLhaus and extracting URL features 

print("\nEDA of URLhaus dataset...\n")

# EDA for raw_urlhause

print(urlhaus_df.columns.tolist())
print(urlhaus_df.head(3))
print(urlhaus_df.shape)
print(urlhaus_df.dtypes)      



print("\nNull URLs:", urlhaus_df['url'].isnull().sum())
print("Duplicate URLs:", urlhaus_df['url'].duplicated().sum())
print("URL status value:", urlhaus_df['url_status'].value_counts())
print("Threat values:", urlhaus_df['threat'].value_counts())


# Data cleaning of URLhaus dataset 

print("\nCleaning URLhaus dataset...")


urlhaus_df = urlhaus_df[['url']].copy()
urlhaus_df = urlhaus_df.dropna()
urlhaus_df = urlhaus_df.drop_duplicates()

print(f"URLhaus clean: {len(urlhaus_df)} rows")


# Extracting URL based malicious features 

def extract_url_features(url):

    try:
        url = str(url).strip()


        # Basic URL features
        url_length = len(url)
        is_https = 1 if url.startswith("https") else 0
        no_of_dots = url.count('.')
        no_of_hyphens = url.count('-')
        no_of_digits = sum(c.isdigit() for c in url)
        no_of_letters = sum(c.isalpha() for c in url)
        no_of_special = len(re.findall(r'[^a-zA-Z0-9.]', url))
        digit_ratio = no_of_digits / url_length if url_length>0 else 0 
        letter_ratio = no_of_letters / url_length if url_length>0 else 0 
        has_obfuscation = 1 if '%' in url else 0 


        # Domain Extraction(Feature extraction) 

        try:
            domain = url.split('/')[2] if '//' in url else url.split('/')[0]
        except:
            domain = url 


        domain_length = len(domain)
        is_domain_ip = is_ip_domain(domain)
        no_of_subdomains = max(0, domain.count('.') -1)
        domain_has_numbers = 1 if re.search(r'\d', domain) else 0
        domain_hyphen_count = domain.count('-')
        domain_word_count = len(domain.split('.'))

        return {
            'URLLength': url_length,
            'IsHTTPS': is_https,
            'NoOfDots': no_of_dots,
            'NoOfHyphensInURL': no_of_hyphens,
            'NoOfDigits': no_of_digits,
            'NoOfLettersInURL': no_of_letters,
            'NoOfSpecialCharsInURL': no_of_special,
            'DigitRatioInURL': digit_ratio,
            'LetterRatioInURL': letter_ratio,
            'HasObfuscation': has_obfuscation,
            'DomainLength': domain_length,
            'IsDomainIP': is_domain_ip,
            'NoOfSubDomain': no_of_subdomains,
            'DomainHasNumbers': domain_has_numbers,
            'DomainHyphenCount': domain_hyphen_count,
            'DomainWordCount': domain_word_count, 
        }
    
    except:
        return None 
    

print("Extracting features from URLhaus URLs...")
print("\nExtracting...\n")

urlhaus_features = urlhaus_df['url'].apply(extract_url_features)
urlhaus_features = urlhaus_features.dropna()
urlhaus_features_df = pd.DataFrame(urlhaus_features.tolist())


# Labling all url as malicious 
urlhaus_features_df['label'] = 1


print(f"URLhaus after feature extraction: {len(urlhaus_features_df)} rows")
print(f"Sample features:\n{urlhaus_features_df.head(3)}")

# print("URLhaus columns:", urlhaus_features_df.columns.tolist())
# print("PhiUSIIL columns:", phiusiil_df.columns.tolist())
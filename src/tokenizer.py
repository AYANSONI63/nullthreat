from collections import Counter
import json 
from pathlib import Path
import numpy as np


# Build Vocabulary 

def build_vocabulary(urls, pad_token="<PAD>", unk_token="<UNK>",):

    char_counter = Counter()

    for url in urls:
        char_counter.update(url)


    characters = sorted(char_counter.keys())

    char2idx = {
        pad_token : 0,
        unk_token : 1,
    }

    for idx, char in enumerate(characters, start=2):

        char2idx[char] = idx

    
    idx2char = {
        idx: char for char, idx in char2idx.items()
    }

    return char2idx, idx2char


# Save vocabulary  

def save_vocabulary(char2idx, idx2char, save_dir):

    save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    
    with open(save_dir / "char2idx.json", "w") as f:
        json.dump(char2idx, f, indent=4)

    
    with open(save_dir / "idx2char", "w") as f:
        json.dump(idx2char, f, indent=4)
    

# Load Vocabulary 

def load_vocabulary(save_dir):

    save_dir = Path(save_dir)

    with open(save_dir / "char2idx.json") as f:
        char2idx = json.load(f)

    with open(save_dir / "idx2char") as f:
        idx2char = json.load(f)

    
    idx2char = {
        int(k): v for k, v in idx2char.items()
    }


    return char2idx, idx2char


# Single URL Encoder 

def encode_url(url, char2idx):

    unk = char2idx["<UNK>"]

    return [
        char2idx.get(char, unk) for char in url
    ]


# Multiple URL Encoder 

def encode_urls(urls, char2idx):

    return [
        encode_url(url, char2idx) for url in urls
    ]


# Padding function 

def pad_sequences(
        sequences,
        max_length=256,
        padding_value=0,
        padding='post',
        truncating='post'
):
    
    padded_sequences = []


    for sequence in sequences:

        # Truncation 

        if len(sequence) > max_length:

            if truncating == 'post':
                sequence = sequence[:max_length]
            
            elif truncating == 'pre':
                sequence = sequence[-max_length:]

            else:
                raise ValueError(
                    "truncating must be either 'pre' or 'post' "
                )
            
        # Padding 

        padding_needed = max_length - len(sequence)

        if padding == "post":

            sequence = (
                sequence + [padding_value] * padding_needed
            )
        
        elif padding == 'pre':

            sequence = (
                [padding_value] * padding_needed + sequence
            )

        else:
            raise ValueError(
                "padding must be either 'pre' or 'post'"
            )
        
        padded_sequences.append(sequence)


    return np.array(padded_sequences, dtype=np.int64)

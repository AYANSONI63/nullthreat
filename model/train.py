import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
import os 
import json



# Loading preprocessed data

print("Loading preprocessed data...")

X_train = np.load("dataset/X_train.npy")
X_test  = np.load("dataset/X_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test  = np.load("dataset/y_test.npy")


print(f"X_train shape : {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train : 0={int((y_train==0).sum())}, 1={int((y_train==1).sum())}")
print(f"y_test : 0={int((y_test==0).sum())}, 1={int((y_test==1).sum())}")


# Building model 

print("Building Model...")

INPUT_SHAPE = X_train.shape[1]

model = tf.keras.Sequential([

    
])
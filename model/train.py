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
    
    # INPUT layer
    tf.keras.layers.Input(shape=(INPUT_SHAPE,)),

    # Block 1
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    # Block 2
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Block 3
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    
    # Output -- getting probability between 0 to 1 
    tf.keras.layers.Dense(1, activation='sigmoid'),

])

model.summary()


# Compiles 

model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)


# Callbacks 


callbacks = [
    
    # Stops training when val_loss stops improving 

    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),


    # Reduce Learning rate when val_loss plateau 

    tf.keras.callbacks.ReduceLROnPlateau(
        moniort='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1 
    ),


    # Saves the best model, checkpoint during training 

    tf.keras.callbacks.ModelCheckpoint(
        filepath="model/saved_model/best_model.keras",
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )

]



# Class weights
# Slight extra penalty for missing malicious URLs


class_weight = {
    0:1.0, # Malicious
    1:1.0  # Safe
}


safe_count = int((y_train == 1).sum())
malicious_count = int((y_train == 0).sum())


if safe_count != malicious_count:
    class_weight[1] = malicious_count/safe_count
    print(f"Classes weight adjusted: {class_weight}")

else:
    print("Classes are balanced - equal weights used")



#  Training 

print("Training model...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=512,
    validation_split=0.1,
    callbacks=callbacks,
    class_weight=class_weight,
    shuffel=True,
    verbose=1,
)

print("\nTraining Complete!")



# Evaluation on test set 

print("\n"+"="*50)
print("EVALUATION ON TEST SET")
print("\n"+"="*50)


# Raw probablities
y_prob = model.predict(X_test, verbose=0).flatten()


# Default thresholds 0.5
y_pred_default = (y_prod >= 0.5).astype(int)


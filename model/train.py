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
        monitor='val_loss',
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
    shuffle=True,
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
y_pred_default = (y_prob >= 0.5).astype(int)



print("\n---Default threshold (0.5)---")
print(classification_report(y_test, y_pred_default, target_names=['Malicious', 'Safe']))


print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_default))


roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.4f}")



# Final optimal threshold from ROC curve 

print("\nFinding optimal threshold...")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)


# Best threshold 

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = float(thresholds[optimal_idx])


print(f"Optimal threshold: {optimal_threshold:.4f}")


# Evaluating at optimal threshold 
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)


print(f"\n--- Optimal Threshold ({optimal_threshold:.4f}) ---")
print(classification_report(y_test, y_pred_optimal,
      target_names=['Malicious', 'Safe']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_optimal))


# Saving best model


print("\nSaving model...")

os.makedirs("model/saved_model", exist_ok=True)

# Save final model
model.save("model/saved_model/nullthreat_model.keras")


# Save threshold — FastAPI will use this to make final decision
threshold_data = {
    "optimal_threshold": optimal_threshold,
    "roc_auc": roc_auc
}

# Saving model and Threshold 

with open("model/saved_model/threshold.json", "w") as f:
    json.dump(threshold_data, f, indent=2)

print("Saved: model/saved_model/nullthreat_model.keras")
print("Saved: model/saved_model/threshold.json")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"ROC AUC        : {roc_auc:.4f}")
print(f"Best threshold : {optimal_threshold:.4f}")
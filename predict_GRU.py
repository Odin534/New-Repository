import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model_path = "C:\\Users\\arkos\\Documents\\Project_Python_Code\\new-repo\\best_model_across_folds.h5"
model = load_model('best_model_across_folds.h5')

# Function to preprocess new data
def preprocess_new_data(file_path):
    # Read the new CSV files
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(columns=["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag",
                          "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time"])
    df_filtered = df[df["New Label"] != 'Rest']
    # Separate features and labels
    X_new = df_filtered.drop("New Label", axis=1).values
    y_new = df_filtered["New Label"].values
    
    # Reshape input to be 3D [samples, timesteps, features]
    X_new = np.reshape(X_new, (X_new.shape[0], 1, X_new.shape[1]))
    
    return X_new, y_new

# Load and preprocess the new data
file_path1 = "C:\\Users\\arkos\\Documents\\Project_Python_Code\\new-repo\\eeg_data_20230801135723.csv"
file_path2 = "C:\\Users\\arkos\\Documents\\Project_Python_Code\\new-repo\\eeg_data_20230801140044.csv"

X_new1, y_new1 = preprocess_new_data(file_path1)
X_new2, y_new2 = preprocess_new_data(file_path2)

# Make predictions
predictions1 = model.predict(X_new1)
predictions2 = model.predict(X_new2)

# Convert the predictions to class labels
class_labels1 = np.argmax(predictions1, axis=1)
class_labels2 = np.argmax(predictions2, axis=1)

print("Shape of X_new1:", X_new1.shape)
print("Shape of class_labels1:", class_labels1.shape)

# Map class labels to defined labels
classification_map = {0: 'Left', 1: 'Right', 2: 'Stop'}
mapped_labels1 = [classification_map[label] for label in class_labels1]
mapped_labels2 = [classification_map[label] for label in class_labels2]



print(f"Predictions for first file: {mapped_labels1}")
print(f"Predictions for second file: {mapped_labels2}")


from sklearn.metrics import accuracy_score

# Step 1: Map the actual labels to your defined labels ('left', 'right', 'stop')
# Ignore the 'Rest' labels
mapped_actual_labels1 = [label for label in y_new1 if label in classification_map.values()]
mapped_actual_labels2 = [label for label in y_new2 if label in classification_map.values()]

# predicted labels
mapped_labels1 = [classification_map[label] for label in class_labels1]
mapped_labels2 = [classification_map[label] for label in class_labels2]

# For the first file
accuracy1 = accuracy_score(mapped_actual_labels1, mapped_labels1)
print(f"Accuracy for the first file: {accuracy1 * 100:.2f}%")

# For the second file
accuracy2 = accuracy_score(mapped_actual_labels2, mapped_labels2)
print(f"Accuracy for the second file: {accuracy2 * 100:.2f}%")


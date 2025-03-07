import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.regularizers import l1_l2
from PyQt5.QtWidgets import QMessageBox
from RobotControl import RobotControl
from PyQt5.QtWidgets import QApplication
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

classification_map = {
    0: 'left',
    1: 'right',
    2: 'stop'
}

robot_controller = RobotControl()

app = QApplication([])
current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']

def load_data(data_dir, processed_files):
    dfs = []
    print("Loading data from directory:", data_dir)  
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and filename not in processed_files:
            print(f"Loading file: {filename}")
            processed_files.append(filename)

            # Read the EEG data from the CSV file
            df = pd.read_csv(os.path.join(data_dir, filename))

            # Remove unwanted columns
            df = df.drop(columns=["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag",
                                  "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time"])
            dfs.append(df)

    # Concatenate all dataframes
    try:
        df_all = pd.concat(dfs)
    except ValueError:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Please add new files to analyze")
        msg.setWindowTitle("No Data Error")
        msg.exec_()
        return None
    print(f"Concatenated all files. Total samples: {len(df_all)}")
    return df_all

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

def normalize_confusion_matrix(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

'''
def undersample_data(X, y):
    # Combine the data into a single dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['label'] = y

    # Get the number of samples in the smallest class
    min_class_size = df['label'].value_counts().min()

    # Resample each class to have the same number of samples as the smallest class
    dfs = []
    for label in df['label'].unique():
        df_class = df[df['label'] == label]
        df_class_resampled = resample(df_class, replace=False, n_samples=min_class_size, random_state=42)
        dfs.append(df_class_resampled)

    # Concatenate the resampled dataframes and shuffle
    df_resampled = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate the features and labels
    X_resampled = df_resampled.drop('label', axis=1).values
    y_resampled = df_resampled['label'].values

    return X_resampled, y_resampled
'''

def balance_data(data, method='undersample'):
    """
    Balance the dataset by undersampling or oversampling.
    
    Parameters:
        - data: DataFrame
        - method: 'undersample' or 'oversample'
        
    Returns:
        - balanced_data: DataFrame
    """
    # Split features and labels
    X = data.drop("New Label", axis=1)
    y = data["New Label"]

    # Combine them for resampling
    data_combined = X.copy()
    data_combined['New Label'] = y

    # Check method and perform balancing
    if method == 'undersample':
        min_class_size = y.value_counts().min()
        dfs = []
        for label in y.unique():
            df_class = data_combined[data_combined['New Label'] == label]
            df_class_resampled = resample(df_class, replace=False, n_samples=min_class_size, random_state=42)
            dfs.append(df_class_resampled)
        balanced_data = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    elif method == 'oversample':
        max_class_size = y.value_counts().max()
        dfs = []
        for label in y.unique():
            df_class = data_combined[data_combined['New Label'] == label]
            df_class_resampled = resample(df_class, replace=True, n_samples=max_class_size, random_state=42)
            dfs.append(df_class_resampled)
        balanced_data = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        raise ValueError("Method should be either 'undersample' or 'oversample'.")

    return balanced_data

def apply_rnn(data):
    # Filter the data for non-'Rest' labels
<<<<<<< HEAD
<<<<<<< HEAD

    df_filtered = data[data["New Label"] != "Rest"]
    #df_filtered = df_filtered[df_filtered["New Label"] != "Forward"] # thi line was added in order to remove MI signal
=======
    df_filtered = data[(data["New Label"] != "Rest") & (data["New Label"] != "Forward")]
>>>>>>> b890af4454623aa514adb85de1e70b85f3118883
=======
    #df_filtered = data[(data["New Label"] != "Rest") & (data["New Label"] != "Forward")]
    df_filtered = data[data["New Label"] != "Forward"]
    #df_filtered = data.copy()
>>>>>>> fd0a7a1e85c8b5f4e053692b62793b12fc5ab411

    # Balance the data using undersampling (or 'oversample' if needed)
    balanced_df = balance_data(df_filtered, method='undersample')
    print("Number of samples after undersampling:")
    print(balanced_df["New Label"].value_counts())

    # Define features and labels
    X = balanced_df.drop("New Label", axis=1).values
    y = balanced_df["New Label"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)  

    # Reshape input to be 3D [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Initialize variable to keep track of the best model and its validation loss
    best_val_loss = float('inf')  # set to positive infinity initially
    best_model = None

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_num = 1

    for train_index, test_index in skf.split(X, y_encoded):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_categorical[train_index], y_categorical[test_index]

        model_path = os.path.join(data_dir, f'best_model_fold_{fold_num}.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential()
            model.add(GRU(100, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2))
            model.add(Dropout(0.5))  # dropout for preventing overfitting
            #model.add(Dense(3, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # Regularization
            model.add(Dense(4, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # Regularization

        # Compile the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("Model Summary:")
        print(model.summary())

        print(f"Started training RNN model with GRU for fold {fold_num}...")

        # Define early stopping and model checkpointing
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

        # using a part of the training set for validation.
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, 
                            callbacks=[early_stopping, checkpoint, TestCallback((X_test, y_test))])

        # Get the best validation loss for this fold
        current_best_val_loss = min(history.history['val_loss'])
        
        # Compare with global best validation loss
        if current_best_val_loss < best_val_loss:
            best_val_loss = current_best_val_loss
            best_model = model

        # Print training and validation loss and accuracy per epoch for this fold
        print(f"Fold {fold_num} - Training Loss per epoch:", history.history['loss'])
        print(f"Fold {fold_num} - Validation Loss per epoch:", history.history['val_loss'])
        print(f"Fold {fold_num} - Training Accuracy per epoch:", history.history['accuracy'])
        print(f"Fold {fold_num} - Validation Accuracy per epoch:", history.history['val_accuracy'])

        # Evaluate the model on the test data for this fold
        _, accuracy = model.evaluate(X_test, y_test)
        print(f"Fold {fold_num} - Test set accuracy: ", accuracy)
        
        # Predict the labels on the test set
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Generate the confusion matrix
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        normalized_conf_matrix = normalize_confusion_matrix(conf_matrix)

        print(f"Fold {fold_num} - Confusion Matrix:")
        print(conf_matrix)

        print(f"Fold {fold_num} - Normalized Confusion Matrix:")
        print(normalized_conf_matrix)

        print(f"Fold {fold_num} - Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

        fold_num += 1

    # Save the best model after all folds
    best_model_path = os.path.join(data_dir, 'best_model_across_folds.h5')
    best_model.save(best_model_path)

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

model_path = os.path.join(data_dir, 'best_model_across_folds.h5')
# Define the real-time classification function for RNN GRU model
def classify_realtime_data_rnn_gru(eeg_data, model_path=model_path):
    
    # Drop unwanted columns from the DataFrame only if they exist
    columns_to_drop = ["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag", 
                       "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time"]

    columns_to_drop = [col for col in columns_to_drop if col in eeg_data.columns]
    eeg_data = eeg_data.drop(columns=columns_to_drop)
    
    # Convert eeg_data to numpy array if it's a DataFrame
    if isinstance(eeg_data, pd.DataFrame):
        eeg_data = eeg_data.values
        
    # Reshape input to be 3D [samples, timesteps, features]
    eeg_data = np.reshape(eeg_data, (eeg_data.shape[0], 1, eeg_data.shape[1]))
    
    # Load the RNN GRU model
    model = load_model(model_path)
    
    # Get the model's predictions for the eeg_data
    predictions = model.predict(eeg_data)
    #RobotControl.command_robot(classification_map[predictions])
    
    # Convert the predictions to class labels
    # This assumes the model's output is a one-hot encoded vector, and we're getting the index of the maximum value as the class label
    class_labels = np.argmax(predictions, axis=1)
    robot_controller.command_robot(class_labels[0])
    return class_labels

# Return the function's signature for review
classify_realtime_data_rnn_gru



if __name__ == "__main__":
    data_dir = os.path.join(current_dir, 'Profile', current_profile)
    processed_files_path = f'{data_dir}/processed_files.json'
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = json.load(f)
    else:
        processed_files = []
    data = load_data(data_dir, processed_files)
    if data is not None:
        apply_rnn(data)
    else:
        print("No new data to analyze.")
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f)
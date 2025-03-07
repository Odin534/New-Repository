import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from PyQt5.QtWidgets import QApplication
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from scipy.stats import skew, kurtosis

app = QApplication([])
current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']

def compute_statistical_features(window_df):
    """
    Compute statistical features for a given window of EEG data.
    
    Parameters:
        - window_df: DataFrame containing EEG data for a single window
        
    Returns:
        - features: DataFrame containing statistical features
    """
    # Filter numeric columns only
    window_df = window_df.select_dtypes(include=[np.number])

    # Compute statistical features
    mean_values = window_df.mean(numeric_only=True)
    var_values = window_df.var(numeric_only=True)
    std_values = window_df.std(numeric_only=True)
    
    # Using apply along with lambda for skew and kurtosis to handle numeric columns
    skew_values = window_df.apply(lambda x: skew(x.dropna()))
    kurt_values = window_df.apply(lambda x: kurtosis(x.dropna()))
    
    # Concatenate all statistical features into a single DataFrame
    features = pd.concat([mean_values, var_values, std_values, skew_values, kurt_values], axis=1)
    features.columns = ['mean', 'var', 'std', 'skew', 'kurt']
    
    return features

def load_data(data_dir, processed_files, window_size=128):
    dfs = []
    print("Loading data from directory:", data_dir)  
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and filename not in processed_files:
            print(f"Loading file: {filename}")
            processed_files.append(filename)

            # Read the EEG data from the CSV file
            df = pd.read_csv(os.path.join(data_dir, filename))


            df = df.drop(columns=["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag",
                                  "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time"])
            features = df.drop(columns=["New Label"])
            labels = df["New Label"]
            
            # Apply 2-second non-overlapping window (128 Hz sampling rate)
            # Each window will contain 256 samples
            num_windows = len(df) // window_size  # This will drop the last few samples if they don't make up a complete window
            
            for i in range(num_windows):
                window_features = features.iloc[i*window_size : (i+1)*window_size]
                window_labels = labels.iloc[i*window_size : (i+1)*window_size]

                # Compute statistical features for this window
                stat_features = compute_statistical_features(window_features)
                
                # Take the most frequent label in this window as the label for the window
                most_frequent_label = window_labels.mode()[0]
                
                # Append the most frequent label to the feature DataFrame
                stat_features["New Label"] = most_frequent_label

                # Append the feature DataFrame to dfs
                dfs.append(stat_features)

  
    df_all = pd.concat(dfs)
    print(f"Concatenated all files. Total samples: {len(df_all)}")
    return df_all

def normalize(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def apply_lstm(data):
    df_filtered = data[(data["New Label"] != "Rest") & (data["New Label"] != "Forward")] 

    # Define features and labels
    X = df_filtered.drop("New Label", axis=1).values
    y = df_filtered["New Label"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y) # One-hot encoding

    # Reshape input to be 3D [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    def balance_data(data, method='undersample'):
        """
        Balance the dataset by undersampling or oversampling.
        
        Parameters:
            - data: DataFrame
            - method: 'undersample' or 'oversample'
            
        Returns:
            - balanced_data: DataFrame
        """
        if method == 'undersample':
            min_samples = min(data["New Label"].value_counts())
            balanced_data = data.groupby("New Label").apply(lambda x: x.sample(min_samples)).reset_index(drop=True)
        elif method == 'oversample':
            max_samples = max(data["New Label"].value_counts())
            balanced_data = data.groupby("New Label").apply(lambda x: x.sample(max_samples, replace=True)).reset_index(drop=True)
        else:
            raise ValueError("Method should be either 'undersample' or 'oversample'")
        
        return balanced_data
    data = balance_data(data, method='undersample')
    print(data['New Label'].value_counts())

    # Split the dataset into train+validation and test set using stratification
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_val_index, test_index in sss.split(X, y.argmax(axis=1)):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]

    # Further split the train+validation set into separate training and validation sets
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # This will split 0.25 of 0.8 = 0.2 for validation
    for train_index, val_index in sss_val.split(X_train_val, y_train_val.argmax(axis=1)):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

    # K-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    all_y_true = []
    all_y_pred = []

    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val.argmax(axis=1)), start=1):
        
        # Check if a model exists for this fold
        model_path = os.path.join(data_dir, f'best_model_fold_{fold_num}.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:       
            # Define the LSTM model with added Dropout, Dense layers and regularization
            model = Sequential()
            model.add(LSTM(100, input_shape=(X_train_val[train_idx].shape[1], X_train_val[train_idx].shape[2])))  
            model.add(Dropout(0.5))  # Dropout layer after LSTM
            model.add(Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))  # Additional Dense layer with regularization
            model.add(Dropout(0.5))  # Dropout layer after Dense layer
            model.add(Dense(3, activation='softmax'))


            # Compile the model
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            print("Model Summary:")
            print(model.summary())

            # Early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)

            # Fit the model with the callback
            model.fit(X_train_val[train_idx], y_train_val[train_idx], epochs=100, batch_size=32, validation_data=(X_train_val[val_idx], y_train_val[val_idx]), callbacks=[early_stopping])
            
            # Save the trained model
            model.save(model_path)
            
            # Evaluate the model on the current fold
            _, accuracy = model.evaluate(X_train_val[val_idx], y_train_val[val_idx])
            fold_accuracies.append(accuracy)

            # Predict the labels
            y_pred = model.predict(X_train_val[val_idx])
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_train_val[val_idx], axis=1)
            all_y_true.extend(y_true_classes)
            all_y_pred.extend(y_pred_classes)

    # Calculate and print the average accuracy over all folds
    avg_accuracy = np.mean(fold_accuracies)
    print("Average accuracy over all folds: ", avg_accuracy)
    print("Model accuracy: ", accuracy)

    
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    normalized_conf_matrix = normalize(conf_matrix)

    print(f"Fold {fold_num} - Confusion Matrix:")
    print(conf_matrix)

    print(f"Fold {fold_num} - Normalized Confusion Matrix:")
    print(normalized_conf_matrix)

    print(f"Fold {fold_num} - Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

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
        apply_lstm(data)
    else:
        print("No new data to analyze.")
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f)

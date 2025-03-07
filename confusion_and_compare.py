import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import welch
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv2D, DepthwiseConv2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, SeparableConv2D, Conv2D

import pywt

# Data loading function
def load_data():
    filepaths = [
        'profile\\Oussama\\eeg_data_20230705150601.csv',
    ]
    print("Loading data...")
    df_list = [pd.read_csv(filepath) for filepath in filepaths]
    print("Data loading complete!")

    return pd.concat(df_list, ignore_index=True)

# Data preprocessing function
def preprocess_data(df):
    # Exclude rows with 'Rest' label
    print("Preprocessing data...")
    df = df[df["New Label"] != "Rest"]

    
    # Encode the labels
    y = LabelEncoder().fit_transform(df["New Label"].values)
    
    # Drop specified columns
    columns_to_drop = ["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag",
                       "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time", "New Label"]
    df = df.drop(columns=columns_to_drop)
    
    # Keep only numeric columns for the feature set
    X = df.select_dtypes(include=[np.number]).values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the FFT feature extraction function
def fft_features(row, fs, stim_freqs, harmonics):
    fft_values = np.abs(np.fft.fft(row))
    fft_freqs = np.fft.fftfreq(len(row), d=1/fs)
    features = []
    for freq in stim_freqs:
        for harmonic in harmonics:
            actual_freq = freq * harmonic
            features.append(fft_values[np.argmin(abs(fft_freqs - actual_freq))])
    return np.array(features)

# Define the training function for FFT-SVM
def train_fft_svm(X, y):
    # Apply FastICA for independent component analysis
    ica = FastICA(n_components=32, random_state=42)
    eeg_data_ica = ica.fit_transform(X)
    
    # Apply the FFT feature extraction
    fs = 128
    stim_freqs = [9, 11, 13]
    harmonics = [1, 2]
    eeg_data_fft = np.apply_along_axis(fft_features, 1, eeg_data_ica, fs, stim_freqs, harmonics)
    
    # Apply PCA for dimensionality reduction
    n_components = min(10, eeg_data_fft.shape[1])
    pca = PCA(n_components=n_components)
    eeg_data_pca = pca.fit_transform(eeg_data_fft)
    
    # Create a SVM classifier with regularization
    clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale'))
    
    # Train the classifier on the whole training set
    clf.fit(eeg_data_pca, y)
    
    # Return the trained classifier and PCA transformer (for use during prediction)
    return clf, pca

# Define the prediction function for FFT-SVM
def predict_fft_svm(clf_pca_tuple, X):
    clf, pca = clf_pca_tuple
    # Apply FastICA for independent component analysis
    ica = FastICA(n_components=32, random_state=42)
    eeg_data_ica = ica.transform(X)
    
    # Apply the FFT feature extraction
    fs = 128
    stim_freqs = [9, 11, 13]
    harmonics = [1, 2]
    eeg_data_fft = np.apply_along_axis(fft_features, 1, eeg_data_ica, fs, stim_freqs, harmonics)
    
    # Apply PCA for dimensionality reduction
    eeg_data_pca = pca.transform(eeg_data_fft)
    
    # Make predictions
    y_pred = clf.predict(eeg_data_pca)
    
    return y_pred

# Displaying the functions
train_fft_svm, predict_fft_svm

# SVM-DWT training function
def train_svm_dwt(X_train, y_train):
    print("Starting DWT Transformation...")
    X_train_dwt = []
    for x in X_train:
        coeffs = pywt.wavedec(x, 'db1', level=4)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        X_train_dwt.append(coeff_arr)
    X_train_dwt = np.array(X_train_dwt)
    print("DWT Transformation Complete!")
    
    print("Starting SVM Training...")
    clf_svm_dwt = SVC(kernel='linear').fit(X_train_dwt, y_train)
    print("SVM Training Complete!")
    return clf_svm_dwt

# SVM-DWT prediction function
def predict_svm_dwt(clf, X_test):
    X_test_dwt = []
    for x in X_test:
        coeffs = pywt.wavedec(x, 'db1', level=4)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        X_test_dwt.append(coeff_arr)
    X_test_dwt = np.array(X_test_dwt)
    return clf.predict(X_test_dwt)

# KNN-FFT training function
def train_knn_fft(X_train, y_train):
    X_train_fft = []
    for x in X_train:
        _, Pxx_den = welch(x, fs=250)
        X_train_fft.append(Pxx_den)
    X_train_fft = np.array(X_train_fft)
    clf_knn_fft = KNeighborsClassifier().fit(X_train_fft, y_train)
    return clf_knn_fft

# KNN-FFT prediction function
def predict_knn_fft(clf, X_test):
    X_test_fft = []
    for x in X_test:
        _, Pxx_den = welch(x, fs=250)
        X_test_fft.append(Pxx_den)
    X_test_fft = np.array(X_test_fft)
    return clf.predict(X_test_fft)

# EEGNet model creation function
def create_EEGNet_model(input_shape, num_classes):
    model = Sequential()

    # First Conv2D layer
    model.add(Conv2D(16, (1, 51), padding='same', input_shape=input_shape, activation='linear', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Depthwise Conv2D layer
    model.add(DepthwiseConv2D((2, 1), use_bias=False, depth_multiplier=2, depthwise_regularizer=l1_l2(l1=1e-4, l2=1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 4)))
    model.add(Dropout(0.25))

    # Separable Conv2D layer
    model.add(SeparableConv2D(16, (1, 15), use_bias=False, padding='same', activation='linear', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 8)))
    model.add(Dropout(0.25))
    
    # Flatten and Dense layer for classification
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_EEGNet(X, y):

    # Reshape input for EEGNet
    X = X[:, :, np.newaxis]

    # One-hot encoding for labels
    y = to_categorical(y)

    # Define the EEGNet model
    model = create_EEGNet_model((X.shape[1], X.shape[2], 1), len(np.unique(y)))

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint('best_model_EEGNet.h5', monitor='val_loss', save_best_only=True)

    # Fit the model
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, 
              callbacks=[early_stopping, checkpoint])

    return model

def predict_EEGNet(model, X):

    # Reshape input for prediction
    X = X[:, :, np.newaxis]
    y_pred = np.argmax(model.predict(X), axis=1)
    
    return y_pred

# Training function for LSTM
def train_LSTM(X, y):
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y = to_categorical(y)
    model = Sequential()
    model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]), dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
    return model

# Prediction function for LSTM
def predict_LSTM(model, X):
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y_pred = np.argmax(model.predict(X), axis=1)
    return y_pred

# Training function for GRU
def train_GRU(X, y):
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y = to_categorical(y)
    model = Sequential()
    model.add(GRU(100, input_shape=(X.shape[1], X.shape[2]), dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
    return model

# Prediction function for GRU
def predict_GRU(model, X):
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y_pred = np.argmax(model.predict(X), axis=1)
    return y_pred

# Main function to run all classifiers and compare their performances
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # SVM-DWT
    print("\nTraining SVM-DWT classifier...")
    clf_svm_dwt = train_svm_dwt(X_train, y_train)
    print("Training complete for SVM-DWT!")

    y_pred_svm_dwt = predict_svm_dwt(clf_svm_dwt, X_test)
    accuracy_svm_dwt = accuracy_score(y_test, y_pred_svm_dwt)
    print(f"Accuracy for SVM-DWT: {accuracy_svm_dwt * 100:.2f}%")

    # KNN-FFT
    print("\nTraining KNN-FFT classifier...")
    clf_knn_fft = train_knn_fft(X_train, y_train)
    print("Training complete for KNN-FFT!")

    y_pred_knn_fft = predict_knn_fft(clf_knn_fft, X_test)
    accuracy_knn_fft = accuracy_score(y_test, y_pred_knn_fft)
    print(f"Accuracy for KNN-FFT: {accuracy_knn_fft * 100:.2f}%")

    # EEGNet
    print("\nTraining EEGNet classifier...")
    model_eegnet = train_EEGNet(X_train, y_train)
    print("Training complete for EEGNet!")

    y_pred_eegnet = predict_EEGNet(model_eegnet, X_test)
    accuracy_eegnet = accuracy_score(y_test, y_pred_eegnet)
    print(f"Accuracy for EEGNet: {accuracy_eegnet * 100:.2f}%")

    # LSTM
    print("\nTraining LSTM classifier...")
    model_lstm = train_LSTM(X_train, y_train)
    print("Training complete for LSTM!")

    y_pred_lstm = predict_LSTM(model_lstm, X_test)
    accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
    print(f"Accuracy for LSTM: {accuracy_lstm * 100:.2f}%")

    # GRU
    print("\nTraining GRU classifier...")
    model_gru = train_GRU(X_train, y_train)
    print("Training complete for GRU!")

    y_pred_gru = predict_GRU(model_gru, X_test)
    accuracy_gru = accuracy_score(y_test, y_pred_gru)
    print(f"Accuracy for GRU: {accuracy_gru * 100:.2f}%")


    print("Starting FFT-SVM Classifier Training...")
    clf_fft_svm, pca_fft = train_fft_svm(X_train, y_train)
    print("FFT-SVM Classifier Training Complete!")
    
    print("Predicting with FFT-SVM Classifier...")
    y_pred_fft_svm = predict_fft_svm((clf_fft_svm, pca_fft), X_test)
    accuracy_fft_svm = accuracy_score(y_test, y_pred_fft_svm)
    print(f"Accuracy for FFT-SVM: {accuracy_fft_svm * 100:.2f}%")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()

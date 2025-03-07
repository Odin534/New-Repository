from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy import signal
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from scipy.signal import welch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import FastICA

import numpy as np
import pandas as pd
import os, json

with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']
#current_profile = profiles[-1] # Get the current profile (assumes the current profile is the last one in the list) 
                                # If the current profile is not the last profile in the list, then get the first profile in the list current_profile = profiles[0]
#data_dir = os.path.join(current_dir, 'Profile', current_profile)


def fft_features(data, fs, freqs, harmonics):
    fft_values = abs(np.fft.rfft(data, n=1024))
    fft_freqs = np.fft.rfftfreq(1024, d=1.0/fs)
    features = []
    for freq in freqs:
        for harmonic in harmonics:
            target_freq = freq * harmonic
            actual_freq = min(fft_freqs, key=lambda x:abs(x-target_freq))
            features.append(fft_values[np.argmin(abs(fft_freqs - actual_freq))])
    return np.array(features)

def preprocess_and_train(data_dir):
    dfs = []
    print("Preprocessing data from directory:", data_dir)
    for filename in os.listdir(data_dir):
        if filename.startswith("eeg"):
            print(f"Processing file: {filename}")

            # Load the data
            data = pd.read_csv(os.path.join(data_dir, filename))
            dfs.append(data)

    # Concatenate all the dataframes into one
    all_data = pd.concat(dfs, ignore_index=True)

    data_non_rest = all_data[all_data['New Label'] != 'Rest']
    
    # Select EEG columns based on the 10-20 system
    eeg_columns = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 
                   'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 
                   'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
    eeg_data = data_non_rest[eeg_columns]
    
    # Apply Butterworth and notch filtering
    fs = 128  # Sampling frequency
    lowcut = 0.5  # Low frequency cutoff (Hz)
    highcut = 60.0  # High frequency cutoff (Hz)
    notch_freq = 50.0  # Notch frequency (Hz)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    notch = notch_freq / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    eeg_data_filtered = signal.lfilter(b, a, eeg_data, axis=0)
    b_notch, a_notch = signal.iirnotch(notch, Q=30.0)
    eeg_data_filtered = signal.lfilter(b_notch, a_notch, eeg_data_filtered, axis=0)

    
    # Print the shape and head of eeg_data for debugging
    print(f"eeg_data shape: {eeg_data.shape}")
    print("eeg_data head:")
    print(eeg_data.head())
    print("ICA and FFT is being performed !")

    ica = FastICA(whiten='unit-variance')
    eeg_data_ica = pd.DataFrame(ica.fit_transform(eeg_data_filtered), columns=eeg_columns)

    # Apply the FFT feature extraction
    fs = 128
    stim_freqs = [9, 11, 13]
    harmonics = [1, 2]
    eeg_data_fft = np.apply_along_axis(fft_features, 1, eeg_data_ica.values, fs, stim_freqs, harmonics)

    # Separate the extracted features according to labels
    labels = data_non_rest['New Label']

    stop_features = eeg_data_fft[labels == "Stop"]
    left_features = eeg_data_fft[labels == "Left"]
    right_features = eeg_data_fft[labels == "Right"]

    stop_labels = labels[labels == "Stop"]
    left_labels = labels[labels == "Left"]
    right_labels = labels[labels == "Right"]

        # Print the shapes of the extracted features and labels
    print("Stop Features Shape:", stop_features.shape)
    print("Left Features Shape:", left_features.shape)
    print("Right Features Shape:", right_features.shape)
    print("Stop Labels Shape:", stop_labels.shape)
    print("Left Labels Shape:", left_labels.shape)
    print("Right Labels Shape:", right_labels.shape)

    # Balance the data
    min_samples = min(stop_features.shape[0], left_features.shape[0], right_features.shape[0])

    # Select a subset of each class to balance the data
    balanced_stop_features = stop_features[:min_samples]
    balanced_left_features = left_features[:min_samples]
    balanced_right_features = right_features[:min_samples]

    # Create balanced labels
    balanced_stop_labels = np.full((min_samples,), 'Stop')
    balanced_left_labels = np.full((min_samples,), 'Left')
    balanced_right_labels = np.full((min_samples,), 'Right')

    # Combine balanced features and labels
    balanced_features = np.concatenate((balanced_stop_features, balanced_left_features, balanced_right_features))
    balanced_labels = np.concatenate((balanced_stop_labels, balanced_left_labels, balanced_right_labels))
    print(balanced_features)
    print(balanced_labels)

    # Standardize the features
    scaler = StandardScaler()

    # Reshape the features to have 2D shape
    standardized_features = scaler.fit_transform(balanced_features.reshape(-1, 1))
    print(standardized_features[:10])

    #mean_std = np.mean(standardized_features), np.std(standardized_features)
    #print("Mean and Standard Deviation of Standardized Features:", mean_std)

    # Perform PCA
    pca = PCA(n_components=standardized_features)
    pca_result = pca.fit_transform(standardized_features)

    # Print explained variance ratio
    #print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(balanced_labels)

    # Apply Random Undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(pca_result, encoded_labels)

    # Split the data into train, validation, and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X_resampled, y_resampled):
        X_train, X_temp = X_resampled[train_index], X_resampled[test_index]
        y_train, y_temp = y_resampled[train_index], y_resampled[test_index]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_index, test_index in sss.split(X_temp, y_temp):
        X_val, X_test = X_temp[val_index], X_temp[test_index]
        y_val, y_test = y_temp[val_index], y_temp[test_index]

    # Train an SVM classifier
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = svm_classifier.predict(X_val)

    # Compute the classification report
    class_names = label_encoder.classes_
    classification_rep = classification_report(y_val, y_pred, target_names=class_names)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Display the normalized confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=normalized_conf_matrix, display_labels=class_names)
    disp.plot(cmap='Blues')

    # Print the classification report
    print(classification_rep)

preprocess_and_train(data_dir)



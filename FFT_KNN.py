from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import FastICA
from scipy import signal
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import os, json

current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    # profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']


# current_profile = profiles[-1] # Get the current profile (assumes the current profile is the last one in the list)
# If the current profile is not the last profile in the list, then get the first profile in the list current_profile = profiles[0]
# data_dir = os.path.join(current_dir, 'Profile', current_profile)

def preprocess_and_train(data_dir):
    dfs = []
    print("Preprocessing data from directory:", data_dir)
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):#filename.startswith("eeg"):
            print(f"Processing file: {filename}")

            # Load the data
            data = pd.read_csv(os.path.join(data_dir, filename))
            dfs.append(data)

    # Concatenate all the dataframes into one
    all_data = pd.concat(dfs, ignore_index=True)

    data_non_rest = all_data[all_data['New Label'] != 'Rest']
    data_non_rest = data_non_rest[data_non_rest['New Label'] != 'Forward']

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
    print("Please be Patient")

    ica = FastICA(whiten='unit-variance')
    eeg_data_ica = pd.DataFrame(ica.fit_transform(eeg_data_filtered), columns=eeg_columns)

    # Define the FFT feature extraction function
    def fft_features(data, fs, freqs, harmonics):
        fft_values = abs(np.fft.rfft(data, n=1024))
        fft_freqs = np.fft.rfftfreq(1024, d=1.0 / fs)
        features = []
        for freq in freqs:
            for harmonic in harmonics:
                target_freq = freq * harmonic
                actual_freq = min(fft_freqs, key=lambda x: abs(x - target_freq))
                features.append(fft_values[np.argmin(abs(fft_freqs - actual_freq))])
        return np.array(features)

    # Apply the FFT feature extraction
    fs = 128
    stim_freqs = [9, 11, 13]
    harmonics = [1, 2]
    eeg_data_fft = np.apply_along_axis(fft_features, 1, eeg_data_ica.values, fs, stim_freqs, harmonics)

    # Apply PCA for dimensionality reduction
    n_components = min(10, eeg_data_fft.shape[1])
    pca = PCA(n_components=n_components)
    eeg_data_pca = pca.fit_transform(eeg_data_fft)

    # Create a StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Get the indices for the training and test sets
    for train_index, test_index in sss.split(eeg_data_pca, data_non_rest['New Label']):
        X_train, X_test = eeg_data_pca[train_index], eeg_data_pca[test_index]
        y_train, y_test = data_non_rest['New Label'].iloc[train_index], data_non_rest['New Label'].iloc[test_index]

    # Create a SVM classifier with regularization
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5 , p = 2, metric='euclidean'))
    print("ICA and FFT is completed !")
    print("Moving on to the classification!")
    print("Please be Patient")
    print("KNN classification Training is going on !")

    # Cross-validation on the training set
    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    print(f"Cross-validation scores: {scores}")
    print(f"Average cross-validation score: {scores.mean()}")

    # Train the classifier on the whole training set
    clf.fit(X_train, y_train)

    # # Save the trained model
    # from joblib import dump
    # model_path = os.path.join(data_dir, 'model_svm.joblib')
    # dump(clf, model_path)

    # Evaluate the classifier on the held-out test set
    y_test_pred = clf.predict(X_test)

    # Print classification report
    print("Test set:")
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    #print('The Confusion Matrix is :\n', cm)
    print('The Confusion Matrix is :\n',cm)
    #Normalized Confusion matrix
    cm_nrm = cm/np.sum(cm,axis=1).reshape(-1,1)
    print('The normalized Confusion Matrix is :\n',cm_nrm)


# Call the function
preprocess_and_train(data_dir)

import math
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.decomposition import FastICA
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import pandas as pd
import os, json
import pywt
from scipy import signal
import pickle

segment_size = 128
current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    # profiles = json.load(file)
current_profile = config['current_profile']
# data_dir = config['data_dir']

label_dict = {0: 'left', 1: 'right', 2: 'stop'}

def load_data():
    data_dir = os.path.join(current_dir, 'Profile', current_profile)
    processed_files_path = f'{data_dir}/processed_files.json'
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = json.load(f)
    else:
        processed_files = []
    dataframes_to_process = []
    print("Loading data from directory:", data_dir)
    for filename in os.listdir(data_dir):
        if (filename.endswith(".csv")) & (filename not in processed_files):
            print(f"Loading file: {filename}")
            processed_files.append(filename)
            df = pd.read_csv(os.path.join(data_dir, filename))
            dataframes_to_process.append(df)
    return dataframes_to_process, processed_files, processed_files_path

def preprocess_and_train(data_list):
    dfs = data_list

    # Concatenate all the dataframes into one
    if(len(dfs)>1):
        all_data = pd.concat(dfs, ignore_index=True)
    else:
        all_data = dfs[0]
    mask = ((all_data['New Label'] != 'Rest') &
            (all_data['New Label'] != 'Forward'))

    # Removing data with label Rest
    data_non_rest = all_data[mask]

    selected_channels = ['PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10']
    all_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5',
                    'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4',
                    'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
    # Select EEG columns based on the 10-20 system
    eeg_columns = selected_channels
    # eeg_columns = ['O1']

    # Extracting only EEG channel data for further processing
    eeg_data = data_non_rest[eeg_columns]

    # Print the shape of eeg_data for debugging
    print(f"eeg_data shape: {eeg_data.shape}")

    # filtration
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

    print("ICA followed by FFT is performed to extract features. PCA used to reduce dimensions of said features")

    ica = FastICA(whiten='unit-variance')
    eeg_data_ica = pd.DataFrame(ica.fit_transform(eeg_data_filtered),
                                columns=eeg_columns)
    eeg_data_ica_with_label = pd.concat((eeg_data_ica, data_non_rest['New Label']), axis=1, join='inner')

    # Implementation of Fast Fourier Transform for Feature Extraction process.
    def fft_features(input_data, fs, target_frequencies, harmonics):
    
        feature_vector_list = []

        i = 0
        while i + segment_size <= len(input_data): # Ensure the segment will be of the correct size
            segment = input_data[i:i + segment_size]
            i += segment_size

            # Get the FFT and frequencies
            fft_values = abs(np.fft.rfft(segment, n=1024))**2  # Square to get power
            fft_freqs = np.fft.rfftfreq(1024, d=1.0/fs)
            
            feature_vector = []

            for freq in target_frequencies:
                for harmonic in harmonics:
                    target_freq = freq * harmonic
                    actual_freq = min(fft_freqs, key=lambda x: abs(x - target_freq))
                    feature_vector.append(fft_values[np.argmin(abs(fft_freqs - actual_freq))])

            feature_vector_list.append(feature_vector)
        return feature_vector_list

    # Apply the FFT feature extraction
    fs = 128
    #target_frequencies = [9, 11, 13]
    #harmonics = [1, 2]
    #eeg_data_fft = np.apply_along_axis(fft_features, 1, eeg_data_ica.values, fs, target_frequencies, harmonics)



    # Function to create the Feature Vector set with labels
    def create_feature_label_vector(input_data):
        eeg_data_ica_with_label = input_data

        # Segregating data according to labels
        data_left = eeg_data_ica_with_label[eeg_data_ica_with_label['New Label'] == 'Left']
        data_right = eeg_data_ica_with_label[eeg_data_ica_with_label['New Label'] == 'Right']
        data_stop = eeg_data_ica_with_label[eeg_data_ica_with_label['New Label'] == 'Stop']

        # FFT section
        feature_vector_left = []
        feature_vector_right = []
        feature_vector_stop = []
        
        fs = 128
        target_frequencies = [9, 11, 13]
        harmonics = [1, 2]
        for channel in eeg_columns:
            feature_vector_left.extend(fft_features(data_left[channel].values, fs, target_frequencies, harmonics))
            feature_vector_right.extend(fft_features(data_right[channel].values, fs, target_frequencies, harmonics))
            feature_vector_stop.extend(fft_features(data_stop[channel].values, fs, target_frequencies, harmonics))

        # Convert the lists to arrays
        feature_vector_left = np.array(feature_vector_left)
        feature_vector_right = np.array(feature_vector_right)
        feature_vector_stop = np.array(feature_vector_stop)

        # Joining the 3 feature sets to make the complete feature set
        feature_vector_matrix = np.concatenate((feature_vector_left, feature_vector_right, feature_vector_stop), axis=0)

        # Flatten the features for the classifier
        feature_vector_set = feature_vector_matrix.reshape(feature_vector_matrix.shape[0], -1)

        n_segments_left = len(feature_vector_left)
        n_segments_right = len(feature_vector_right)
        n_segments_stop = len(feature_vector_stop)

        return feature_vector_set, np.hstack((np.zeros(n_segments_left),
                                            np.ones(n_segments_right),
                                            np.ones(n_segments_stop) * 2))

    # Create the Feature Vector set
    feature_vector_set, label = create_feature_label_vector(eeg_data_ica_with_label)

    if feature_vector_set.shape[0] != label.shape[0]:
        min_samples = min(feature_vector_set.shape[0], label.shape[0])
        feature_vector_set = feature_vector_set[:min_samples]
        label = label[:min_samples]

    # Apply Principal Component Analysis for dimensionality reduction on the Feature Set
    n_components = min(10, feature_vector_set.shape[1])
    pca = PCA(n_components=n_components)
    eeg_data_pca = pca.fit_transform(feature_vector_set)

    assert feature_vector_set.shape[0] == label.shape[0], "Inconsistent number of samples between feature_vector_set and label"


        # Split the Feature Set into training, validation, and test sets
    X, X_test, y, y_test = train_test_split(eeg_data_pca, label, test_size=0.33, random_state=42)

    data_dir = os.path.join(current_dir, 'Profile', current_profile)
    saved_model_path = f'{data_dir}/latest_model.sav'

    if os.path.exists(saved_model_path):
        trained_model = pickle.load(open(saved_model_path, 'rb'))

        scores = cross_val_score(trained_model, X, y, cv=5, n_jobs=-1)
        print(f"Cross-validation scores: {scores}")
        print(f"Average cross-validation score: {scores.mean()}")

        trained_model.fit(X, y)
        filename = f'{data_dir}/latest_model.sav'
        pickle.dump(trained_model,open(filename, 'wb'))

        y_test_pred = trained_model.predict(X_test)
        c_m = confusion_matrix(y_test, y_test_pred)
        test_samples_left, test_samples_right, test_samples_stop = np.sum(c_m, axis=1)
        print("Test set:", X_test.shape)
        print("left_samples:", test_samples_left)
        print("right_samples:", test_samples_right)
        print("stop_samples:", test_samples_stop)
        print(c_m)

        # RobotControl.command_robot(y_test_pred)
        # Print classification report

        print("Classification report on new data", classification_report(y_test, y_test_pred))
    else:
        # Create a SVM classifier with regularization
        clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale'))
        print("ICA for Artifact removal and DWT for feature extraction is completed !")
        print("SVM classifier Training is going on !")

        # Cross-validation on the training set
        scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        print(f"Cross-validation scores: {scores}")
        print(f"Average cross-validation score: {scores.mean()}")

        # Train the classifier on the whole training set
        clf.fit(X, y)
        filename = f'{data_dir}/latest_model.sav'
        pickle.dump(clf,open(filename, 'wb'))

        # Evaluate the classifier on the held-out test set
        y_test_pred = clf.predict(X_test)
        c_m = confusion_matrix(y_test, y_test_pred, normalize='true')
        test_samples_left, test_samples_right, test_samples_stop = np.sum(c_m, axis=1)
        print("Test set:", X_test.shape)
        print("left_samples:", test_samples_left)
        print("right_samples:", test_samples_right)
        print("stop_samples:", test_samples_stop)
        print(c_m)

        # RobotControl.command_robot(y_test_pred)
        # Print classification report

        print('Initial training report', classification_report(y_test, y_test_pred))


    dataframe_list, processed_files, processed_files_path = load_data()

    if len(dataframe_list) > 0:
        preprocess_and_train(dataframe_list)
    else:
        print("No new data to analyze.")
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f)

    preprocess_and_train(data_list)

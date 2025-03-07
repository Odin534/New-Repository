import pandas as pd
import numpy as np
from scipy.signal import welch
import os, json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from scipy import signal
from joblib import dump
from scipy.signal import butter, filtfilt, iirnotch
from joblib import load

# Channels to visualize
channels = ['P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4']

'''
channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 
                   'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 
                   'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
'''
current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']

# Define the stimulus frequencies for each label
stimulus_freqs = {
    'Stop': 9,
    'Left': 11,
    'Right': 13
}

fs=128

def load_processed_files(processed_files_path):
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as file:
            return json.load(file)
    else:
        return []

def save_processed_files(processed_files, processed_files_path):
    with open(processed_files_path, 'w') as file:
        json.dump(processed_files, file)



def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, notch_freq, fs, Q=30):
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, Q)
    return filtfilt(b, a, data)


# Applying the filters to the EEG data
def filter_eeg_data(data):
    for channel in channels:
        data[channel] = bandpass_filter(data[channel], 0.5, 60, fs)
        data[channel] = notch_filter(data[channel], 50, fs)
    return data

'''

    # Applying ICA for artifact removal
    eeg_columns = channels  
    ica = FastICA(whiten='unit-variance')
    eeg_data_ica = pd.DataFrame(ica.fit_transform(data[eeg_columns]), columns=eeg_columns)
    data[eeg_columns] = eeg_data_icaS

    # Apply ICA for artifact removal
    ica = FastICA(n_components=len(channels))
    data_transformed = ica.fit_transform(data[channels])
    data_transformed_df = pd.DataFrame(data_transformed, columns=channels)
    data[channels] = data_transformed_df

    return data

'''


def preprocess_and_train(data_dir):

    data_dir = os.path.join(current_dir, 'Profile', current_profile)
    processed_files_path = f'{data_dir}/processed_files.json'
    processed_files = load_processed_files(processed_files_path)

    new_files = [f for f in os.listdir(data_dir) if f.startswith("eeg") and f not in processed_files]

    if not new_files:
        print("No new files to analyze")
        return
    
    dfs = []
    print("Preprocessing data from directory:", data_dir)
    for filename in new_files:
        print(f"Processing file: {filename}")

        # Load the data
        data = pd.read_csv(os.path.join(data_dir, filename))
        data = filter_eeg_data(data)
        dfs.append(data)
        processed_files.append(filename)

    # Concatenate the new EEG data files
    eeg_data_new_combined = pd.concat(dfs, ignore_index=True)
    # Check the shape of the combined dataset
    print(eeg_data_new_combined.shape)
    
    save_processed_files(processed_files, processed_files_path)

    def plot_psd_for_all_channels(data, label):
        plt.figure(figsize=(20, 15))
        
        # Iterate through each channel and plot its PSD
        for idx, channel in enumerate(channels, 1):
            plt.subplot(5, 2, idx)  # 5 rows and 2 columns of subplots
            filtered_data = data[data['New Label'] == label]
            frequencies, psd_values = welch(filtered_data[channel], fs, nperseg=256)
            plt.plot(frequencies, 10 * np.log10(psd_values))
            plt.axvline(stimulus_freqs[label], color='red', linestyle='--', label=f'Stimulus Frequency: {stimulus_freqs[label]}Hz')
            plt.xlim([0, 40])  # Limiting to 40Hz for better visibility of our frequencies of interest
            plt.title(f'Channel: {channel}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'Power Spectral Density (PSD) for {label} across all channels', y=1.05)
        plt.show()

        # Plot the PSD for "Stop" label across all channels
        plot_psd_for_all_channels(eeg_data_new_combined, 'Stop')
        # Plot the PSD for "Left" label across all channels
        plot_psd_for_all_channels(eeg_data_new_combined, 'Left')

        # Plot the PSD for "Right" label across all channels
        plot_psd_for_all_channels(eeg_data_new_combined, 'Right')

    def compute_fft(data, fs=128):
        N = len(data)
        fft_values = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(N, 1.0/fs)
        return frequencies, np.abs(fft_values)

    # Define function to extract features from power spectrum
    def extract_features(data, label, channel, fs=128):
        # Filter data for the given label
        filtered_data = data[data['New Label'] == label]
        
        # Compute power spectrum using Welch's method (PSD features)
        frequencies_psd, psd_values = welch(filtered_data[channel], fs=fs, nperseg=256)

        # Compute FFT values using the compute_fft function
        frequencies_fft, fft_values = compute_fft(filtered_data[channel], fs=fs)
    
        
        # Compute FFT values
        frequencies_fft, fft_values = np.fft.rfftfreq(len(filtered_data[channel]), 1.0/fs), np.abs(np.fft.rfft(filtered_data[channel]))
        
        # Helper function to extract peak amplitude and harmonics for given values and frequencies
        def extract_peak_harmonics(frequencies, values):
            stimulus_freq = stimulus_freqs[label]
            peak_amplitude = values[np.where(frequencies == stimulus_freq)][0] if stimulus_freq in frequencies else 0
            
            harmonic_peaks = []
            for i in range(2, 6):  # From 2nd to 5th harmonic
                harmonic_freq = stimulus_freq * i
                harmonic_amplitude = values[np.where(frequencies == harmonic_freq)][0] if harmonic_freq in frequencies else 0
                harmonic_peaks.append(harmonic_amplitude)
            
            return peak_amplitude, harmonic_peaks
        
        # Extract features for PSD and FFT
        peak_amplitude_psd, harmonic_peaks_psd = extract_peak_harmonics(frequencies_psd, psd_values)
        peak_amplitude_fft, harmonic_peaks_fft = extract_peak_harmonics(frequencies_fft, fft_values)
        
        return {
            'Peak Amplitude PSD': peak_amplitude_psd,
            '2nd Harmonic PSD': harmonic_peaks_psd[0],
            '3rd Harmonic PSD': harmonic_peaks_psd[1],
            '4th Harmonic PSD': harmonic_peaks_psd[2],
            '5th Harmonic PSD': harmonic_peaks_psd[3],
            'Peak Amplitude FFT': peak_amplitude_fft,
            '2nd Harmonic FFT': harmonic_peaks_fft[0],
            '3rd Harmonic FFT': harmonic_peaks_fft[1],
            '4th Harmonic FFT': harmonic_peaks_fft[2],
            '5th Harmonic FFT': harmonic_peaks_fft[3],
            'Channel': channel,
            'Label': label
        }


    # Extract features for each channel and label
    features_data = []

    for channel in channels:
        for label in ['Stop', 'Left', 'Right']:
            feature_data = extract_features(eeg_data_new_combined, label, channel)
            features_data.append(feature_data)

    # Convert to DataFrame for better visualization
    features_df = pd.DataFrame(features_data)


    #print(features_df)

    # Extract numerical features
    numerical_features = [
    'Peak Amplitude PSD', '2nd Harmonic PSD', '3rd Harmonic PSD', '4th Harmonic PSD', '5th Harmonic PSD',
    'Peak Amplitude FFT', '2nd Harmonic FFT', '3rd Harmonic FFT', '4th Harmonic FFT', '5th Harmonic FFT'
    ]
    X = features_df[numerical_features]

    # Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_standardized)

    # Calculate explained variance ratio for each principal component
    explained_variance_ratio = pca.explained_variance_ratio_

    # Visualize the explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='Individual Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio of Principal Components')
    plt.legend()
    plt.tight_layout()
    plt.show()

    explained_variance_ratio.cumsum()

    from sklearn.model_selection import train_test_split

    # Extract first three principal components
    X_transformed = X_pca[:, :3]

    # Extract labels
    y = features_df['Label']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel='linear', decision_function_shape='ovr')  # 'ovr' stands for one-vs-rest
    svm_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    accuracy, classification_rep

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # normalize

    # Print normalized confusion matrix to terminal
    print("Normalized Confusion Matrix:")
    print(normalized_conf_matrix)
    
    # Plotting confusion matrix
    plt.figure(figsize=(12, 5))

    # Classification confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=svm_classifier.classes_, yticklabels=svm_classifier.classes_)
    plt.title('Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(normalized_conf_matrix, annot=True, cmap='Blues', fmt='.2f', xticklabels=svm_classifier.classes_, yticklabels=svm_classifier.classes_)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()

    # Determine the minimum class size
    min_class_size = y.value_counts().min()

    # Perform undersampling for each class
    undersampled_data = []

    for label in ['Stop', 'Left', 'Right']:
        label_data = features_df[features_df['Label'] == label]
        undersampled_data.append(label_data.sample(min_class_size, random_state=42))

    # Combine the undersampled data
    undersampled_df = pd.concat(undersampled_data, ignore_index=True)

    # Check the class distribution after undersampling
    class_distribution = undersampled_df['Label'].value_counts()

    class_distribution

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

    # Features and target variable
    X_undersampled = undersampled_df[numerical_features]
    y_undersampled = undersampled_df['Label']

    # Split data into training set and temporary set
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X_undersampled, y_undersampled, test_size=0.2, stratify=y_undersampled, random_state=42)

    # Further split the temporary set into test and validation sets
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Perform stratified k-fold cross-validation on the training set
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(svm_classifier, X_train_temp, y_train_temp, cv=kfold, scoring='accuracy')

    # Check if an existing model exists
    model_path = os.path.join(data_dir, 'best_model_svm.joblib')
    if os.path.exists(model_path):
        print("Loading existing model...")
        svm_classifier = load(model_path)
        # Split combined data for evaluation and testing
        X_train, X_test, y_train, y_test = train_test_split(X_undersampled, y_undersampled, test_size=0.2, random_state=42)
        
        # Calculate cross-validation score on the training data
        cross_val_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)
        print(f"Cross-validation Score: {cross_val_scores.mean():.2f} (+/- {cross_val_scores.std() * 2:.2f})")
        
        # Calculate test accuracy
        test_accuracy = svm_classifier.score(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.2f}")
    else:
        # Initialize SVM classifier if no existing model
        svm_classifier = SVC(kernel='linear', decision_function_shape='ovr')

    # Train the classifier on the entire training set and predict accuracy on the test set
    svm_classifier.fit(X_train_temp, y_train_temp)
    test_accuracy = svm_classifier.score(X_test, y_test)

    # Save the trained model
    model_path = os.path.join(data_dir, 'best_model_svm.joblib')
    dump(svm_classifier, model_path)

    cross_val_scores.mean(), test_accuracy
    print(cross_val_scores.mean(), test_accuracy)

    # SVM Model details
    svm_model_details = svm_classifier.get_params()

    # Compute normalized confusion matrix
    conf_matrix_test = confusion_matrix(y_test, svm_classifier.predict(X_test))
    normalized_conf_matrix_test = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]

    print(svm_model_details)
    

preprocess_and_train(data_dir)
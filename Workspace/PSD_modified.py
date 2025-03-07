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
# from scipy import signal
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

    # Define function to extract features from power spectrum
    def extract_features(data, label, channel, fs=128, segment_length=128):
        # Filter data for the given label
        filtered_data = data[data['New Label'] == label][channel]
        
        # Create overlapping segments
        num_segments = len(filtered_data) // (segment_length // 2) - 1
        segments = [filtered_data[i:i+segment_length] for i in range(0, len(filtered_data) - segment_length, segment_length // 2)]
        
        # Extract features for each segment
        features_list = []
        for segment in segments:
            # Compute power spectrum using Welch's method
            frequencies, psd_values = welch(segment, fs=fs, nperseg=segment_length)
            
            # Extract peak amplitude at the stimulus frequency
            stimulus_freq = stimulus_freqs[label]
            peak_amplitude = psd_values[np.where(frequencies == stimulus_freq)][0] if stimulus_freq in frequencies else 0
            
            # Identify and extract harmonic peaks
            harmonic_peaks = []
            for i in range(2, 6):  # From 2nd to 5th harmonic
                harmonic_freq = stimulus_freq * i
                harmonic_amplitude = psd_values[np.where(frequencies == harmonic_freq)][0] if harmonic_freq in frequencies else 0
                harmonic_peaks.append(harmonic_amplitude)
            
            # Add to features list
            features_list.append([peak_amplitude] + harmonic_peaks)
        
        return features_list


    # Extract features for each channel and label
    features_data = []

    for channel in channels:
        for label in ['Stop', 'Left', 'Right']:
            features_list = extract_features(eeg_data_new_combined, label, channel)
            for feature_set in features_list:
                peak_amplitude = feature_set[0]
                harmonic_peaks = feature_set[1:]
                features_data.append({
                    'Channel': channel,
                    'Label': label,
                    'Peak Amplitude': peak_amplitude,
                    '2nd Harmonic': harmonic_peaks[0],
                    '3rd Harmonic': harmonic_peaks[1],
                    '4th Harmonic': harmonic_peaks[2],
                    '5th Harmonic': harmonic_peaks[3]
            })

    # Convert to DataFrame for better visualization
    features_df = pd.DataFrame(features_data)

    features_df

    # Extract numerical features
    numerical_features = ['Peak Amplitude', '2nd Harmonic', '3rd Harmonic', '4th Harmonic', '5th Harmonic']
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
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.svm import SVC
    from imblearn.under_sampling import RandomUnderSampler


    # Extract first three principal components
    X_transformed = X_pca[:, :3]

    # Extract labels
    y = features_df['Label']

    # Initial train-test split (80% train and validation, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

    # Further split the training data into train and validation sets (81.25% train, 18.75% validation of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1875, random_state=42, stratify=y_temp)
    
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)

    # Apply RandomUnderSampler on training data
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Count the number of occurrences of each class label after undersampling
    unique_elements, counts_elements = np.unique(y_train_resampled, return_counts=True)
    print("Frequency of each class label after undersampling:")
    print(np.asarray((unique_elements, counts_elements)))
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'decision_function_shape': ['ovo', 'ovr']
    }

    # Initialize GridSearchCV and fit it to the training data
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print(f"Best parameters from GridSearchCV: {best_params}")

    # Initialize and train the SVM classifier with the best parameters
    svm_classifier = SVC(**best_params)
    svm_classifier.fit(X_train_resampled, y_train_resampled)

    # Validate the model on the validation set (Optional)
    y_val_pred = svm_classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy}")

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:", classification_rep)
    print("Test Accuracy:", accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=svm_classifier.classes_)
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Print normalized confusion matrix to terminal
    print("Normalized Confusion Matrix:")
    print(normalized_conf_matrix)

preprocess_and_train(data_dir)
    
    
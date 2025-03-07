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
    






'''
    # Apply Butterworth and notch filtering
    lowcut = 0.5  # Low frequency cutoff (Hz)
    highcut = 60.0  # High frequency cutoff (Hz)
    notch_freq = 50.0  # Notch frequency (Hz)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    notch = notch_freq / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    #eeg_data_filtered = signal.lfilter(b, a, eeg_data_new_combined, axis=0)
    numeric_cols = eeg_data_new_combined.select_dtypes(include=[np.number]).columns.tolist()
    eeg_data_numeric = eeg_data_new_combined[numeric_cols].to_numpy()
    eeg_data_filtered = signal.lfilter(b, a, eeg_data_numeric, axis=0)

    b_notch, a_notch = signal.iirnotch(notch, Q=30.0)
    #eeg_data_cleaned = signal.lfilter(b_notch, a_notch, eeg_data_filtered, axis=0)
    eeg_data_cleaned = signal.lfilter(b_notch, a_notch, eeg_data_filtered, axis=0)

   
# Load the first file
eeg_data_1 = pd.read_csv('profile\\Subhodeep\\eeg_data_20230705142209.csv')

# Display the first few rows
eeg_data_1.head()

# Load the remaining EEG data files
eeg_data_2 = pd.read_csv('profile\\Subhodeep\\eeg_data_20230705142827.csv')
eeg_data_3 = pd.read_csv('profile\\Subhodeep\\eeg_data_20230705143714.csv')

# Load the hybrid data files
hybrid_data_1 = pd.read_csv('profile\\Subhodeep\\hybrid_data_20230804164650.csv')
hybrid_data_2 = pd.read_csv('profile\\Subhodeep\\hybrid_data_20230804165122.csv')


# Concatenate the EEG data files
eeg_data_combined = pd.concat([eeg_data_1, eeg_data_2, eeg_data_3], ignore_index=True)

# Concatenate the hybrid data files
hybrid_data_combined = pd.concat([hybrid_data_1, hybrid_data_2], ignore_index=True)

# Check the shape of the combined datasets
eeg_data_combined.shape, hybrid_data_combined.shape

# Get the distribution of unique labels for both datasets
eeg_label_distribution = eeg_data_combined['New Label'].value_counts()
hybrid_label_distribution = hybrid_data_combined['New Label'].value_counts()

eeg_label_distribution, hybrid_label_distribution
# Plotting the EEG data for the specified channels
plt.figure(figsize=(20, 12))

# Plot each channel
for channel in channels:
    plt.plot(eeg_data_combined[channel][:5000], label=channel)  # Plotting the first 5000 rows for clarity

plt.title('EEG Data for Specified Channels (First 5000 rows)')
plt.xlabel('Time (samples)')
plt.ylabel('EEG Signal Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plotting the Hybrid data for the specified channels
plt.figure(figsize=(20, 12))

# Plot each channel
for channel in channels:
    plt.plot(hybrid_data_combined[channel][:5000], label=channel)  # Plotting the first 5000 rows for clarity

plt.title('Hybrid Data for Specified Channels (First 5000 rows)')
plt.xlabel('Time (samples)')
plt.ylabel('EEG Signal Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

from scipy.signal import welch
import numpy as np

# Define the stimulus frequencies for each label
stimulus_freqs = {
    'Stop': 9,
    'Left': 11,
    'Right': 13
}

# Function to compute and plot PSD for a given label and dataset
def plot_psd_for_label(data, label, channel):
    # Filter data for the given label
    filtered_data = data[data['New Label'] == label]
    
    # Compute PSD
    frequencies, psd_values = welch(filtered_data[channel], fs=256, nperseg=512)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, 10 * np.log10(psd_values), label=f'{label} - {channel}')
    plt.axvline(stimulus_freqs[label], color='red', linestyle='--', label=f'Stimulus Frequency: {stimulus_freqs[label]}Hz')
    plt.xlim([0, 40])  # Limiting to 40Hz for better visibility of our frequencies of interest
    plt.title(f'Power Spectral Density (PSD) for {label} - {channel}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the PSD for "Stop" label in EEG data
plot_psd_for_label(eeg_data_combined, 'Stop', 'Pz')
# Plot the PSD for "Left" label in EEG data
plot_psd_for_label(eeg_data_combined, 'Left', 'Pz')

# Plot the PSD for "Right" label in EEG data
plot_psd_for_label(eeg_data_combined, 'Right', 'Pz')

'''

'''


# Define the stimulus frequencies for each label
stimulus_freqs = {
    'Stop': 9,
    'Left': 11,
    'Right': 13
}

# Load the newly uploaded EEG data files
eeg_data_new_1 = pd.read_csv('profile\Oussama\eeg_data_20230705150601.csv')
eeg_data_new_2 = pd.read_csv('profile\Oussama\eeg_data_20230705151559.csv')
eeg_data_new_3 = pd.read_csv('profile\Oussama\eeg_data_20230705151932.csv')
eeg_data_new_4 = pd.read_csv('profile\Oussama\eeg_data_20230705152259.csv')
eeg_data_new_5 = pd.read_csv('profile\Oussama\eeg_data_20230728150927.csv')
eeg_data_new_6 = pd.read_csv('profile\Oussama\eeg_data_20230728151256.csv')
eeg_data_new_7 = pd.read_csv('profile\Oussama\eeg_data_20230728151642.csv')
eeg_data_new_8 = pd.read_csv('profile\Oussama\eeg_data_20230728152219.csv')
eeg_data_new_9 = pd.read_csv('profile\Oussama\eeg_data_20230728160419.csv')
eeg_data_new_10 = pd.read_csv('profile\Oussama\eeg_data_20230728160845.csv')


# Concatenate the new EEG data files
eeg_data_new_combined = pd.concat([eeg_data_new_1, eeg_data_new_2, eeg_data_new_3, 
                                  eeg_data_new_4, eeg_data_new_5, eeg_data_new_6, eeg_data_new_7, eeg_data_new_8, eeg_data_new_9,eeg_data_new_10], ignore_index=True)

# Check the shape of the combined dataset
eeg_data_new_combined.shape




def plot_psd_for_all_channels(data, label):
    plt.figure(figsize=(20, 15))
    
    # Iterate through each channel and plot its PSD
    for idx, channel in enumerate(channels, 1):
        plt.subplot(5, 2, idx)  # 5 rows and 2 columns of subplots
        filtered_data = data[data['New Label'] == label]
        frequencies, psd_values = welch(filtered_data[channel], fs=256, nperseg=512)
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
def extract_features(data, label, channel, fs=256):
    # Filter data for the given label
    filtered_data = data[data['New Label'] == label]
    
    # Compute power spectrum using Welch's method
    frequencies, psd_values = welch(filtered_data[channel], fs=fs, nperseg=512)
    
    # Extract peak amplitude at the stimulus frequency
    stimulus_freq = stimulus_freqs[label]
    peak_amplitude = psd_values[np.where(frequencies == stimulus_freq)][0] if stimulus_freq in frequencies else 0
    
    # Identify and extract harmonic peaks (up to 4 harmonics considered for simplicity)
    harmonic_peaks = []
    for i in range(2, 6):  # From 2nd to 5th harmonic
        harmonic_freq = stimulus_freq * i
        harmonic_amplitude = psd_values[np.where(frequencies == harmonic_freq)][0] if harmonic_freq in frequencies else 0
        harmonic_peaks.append(harmonic_amplitude)
    
    return peak_amplitude, harmonic_peaks

# Extract features for each channel and label
features_data = []

for channel in channels:
    for label in ['Stop', 'Left', 'Right']:
        peak_amplitude, harmonic_peaks = extract_features(eeg_data_new_combined, label, channel)
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # normalize

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

# Train the classifier on the entire training set and predict accuracy on the test set
svm_classifier.fit(X_train_temp, y_train_temp)
test_accuracy = svm_classifier.score(X_test, y_test)

cross_val_scores.mean(), test_accuracy
print(cross_val_scores.mean(), test_accuracy)

# SVM Model details
svm_model_details = svm_classifier.get_params()

# Compute normalized confusion matrix
conf_matrix_test = confusion_matrix(y_test, svm_classifier.predict(X_test))
normalized_conf_matrix_test = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]


print(svm_model_details)

'''
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import FastICA
from scipy import signal
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import os, json

current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']
#current_profile = profiles[-1] # Get the current profile (assumes the current profile is the last one in the list) 
                                # If the current profile is not the last profile in the list, then get the first profile in the list current_profile = profiles[0]
#data_dir = os.path.join(current_dir, 'Profile', current_profile)

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

    #ica = FastICA(whiten='unit-variance')
    #eeg_data_ica = pd.DataFrame(ica.fit_transform(eeg_data_filtered), columns=eeg_columns)
    

    def extract_features(data, fs, window_size=256, overlap=128, freqs=None, harmonics=None):
        num_windows = (data.shape[0] - window_size) // overlap + 1
        features = []
        for i in range(num_windows):
            start = i * overlap
            end = start + window_size
            window = data[start:end]

            fft_features = []
            for freq in freqs:
                for harmonic in harmonics:
                    target_freq = freq * harmonic
                    fft_values = abs(np.fft.rfft(window, n=1024))
                    fft_freqs = np.fft.rfftfreq(1024, d=1.0/fs)
                    actual_freq = min(fft_freqs, key=lambda x:abs(x-target_freq))
                    fft_features.append(fft_values[np.argmin(abs(fft_freqs - actual_freq))])

            features.append(fft_features)
        
        return np.array(features)
    
    # Apply the FFT feature extraction
    fs = 128
    stim_freqs = [9, 11, 13]
    harmonics = [1, 2]
    eeg_data_fft = extract_features(eeg_data_filtered, fs, freqs=stim_freqs, harmonics=harmonics)
    # Flatten the features for each window
    eeg_data_flattened = eeg_data_fft.reshape(eeg_data_fft.shape[0], -1)
    # Convert to DataFrame for better visualization
    columns = [f'Freq_{freq}_Harm_{harm}' for freq in stim_freqs for harm in harmonics]
    features_df = pd.DataFrame(eeg_data_flattened, columns=columns)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(features_df)



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

    print("Cumulative Explained Variance Ratio:")
    print(explained_variance_ratio.cumsum())

    # Create a StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Get the indices for the training and test sets
    for train_index, test_index in sss.split(X_pca, data_non_rest['New Label']):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = data_non_rest['New Label'].iloc[train_index], data_non_rest['New Label'].iloc[test_index]

    # Create a SVM classifier with regularization
    clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale'))
    print("ICA and FFT is completed !")
    print("Moving on to the classification!")
    print("SVM classification Training is going on !")

    # Cross-validation on the training set
    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    print(f"Cross-validation scores: {scores}")
    print(f"Average cross-validation score: {scores.mean()}")

    # Train the classifier on the whole training set
    clf.fit(X_train, y_train)

    # Save the trained model
    from joblib import dump
    model_path = os.path.join(data_dir, 'model_svm.joblib')
    dump(clf, model_path)
    
    # Evaluate the classifier on the held-out test set
    y_test_pred = clf.predict(X_test)

    # Compute the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_test_pred)
    
    # Normalize the confusion matrix
    confusion_mat_normalized = confusion_matrix(y_test, y_test_pred, normalize='true')
    
    class_names = sorted(y_test.unique())  

    # Print regular confusion matrix
    print("\nConfusion Matrix:")
    print("\t", "\t".join(class_names))
    for true_label, row in zip(class_names, confusion_mat):
        print(true_label, "\t", "\t".join(map(str, row)))
        
    print(confusion_mat)

    # Print normalized confusion matrix
    print("\nNormalized Confusion Matrix:")
    print("\t", "\t".join(class_names))
    for true_label, row in zip(class_names, confusion_mat_normalized):
        print(true_label, "\t", "\t".join(['{:.2f}'.format(val) for val in row]))

    # Print classification report
    print("Test set:")
    print(classification_report(y_test, y_test_pred))
    
'''
from joblib import load
model_path = os.path.join(data_dir, 'model_svm.joblib')
def classify_realtime_data(eeg_data, model_path=model_path):
    # Apply FFT
    eeg_data_fft = abs(np.fft.rfft(eeg_data, axis=0))
    # Apply ICA
    transformer = FastICA(n_components=10, random_state=0)
    eeg_data_ica = transformer.fit_transform(eeg_data_fft)
    
    # 2. PCA Reduction
    n_components = min(10, eeg_data_ica.shape[1])
    pca = PCA(n_components=n_components)
    eeg_data_pca = pca.fit_transform(eeg_data_ica)
    # 3. Load SVM model and classify
    clf = load(model_path)
    predictions = clf.predict(eeg_data_pca)
    return predictions
'''

# Call the function
preprocess_and_train(data_dir)



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
from RobotControl import RobotControl
# from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from joblib import load

fs=128

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

channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 
                   'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 
                   'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

robot_controller = RobotControl()

def real_time_classification_SVM(new_data_point, channels = channels, fs=128):
    # Assuming new_data_point is a DataFrame containing new EEG data


    
    # Preprocess the new data
    new_data_point = filter_eeg_data(new_data_point)
    
    # Feature extraction logic needed here, but assuming a function like extract_features
    # is defined elsewhere in your code, and it returns a structured data (e.g., array) of features:
    features_list = []
    for channel in channels:
        features_list.extend(extract_features(new_data_point, channel))
    
    features = np.array(features_list).reshape(1, -1)  # Make sure features is a 2D array
    
    # Load the trained model
    svm_classifier = load('best_svm_model.joblib')
    
    # Perform prediction
    prediction = svm_classifier.predict(features)
    
    # Translate numeric predictions back to class labels if needed
    label_mapping = {
        0: 'stop',
        1: 'left',
        2: 'right',
        3: 'forward'
    }
    
    predicted_label = label_mapping[prediction[0]]

    # Send the command to the robot
    robot_controller.command_robot(predicted_label)
    
    return predicted_label

if __name__ == "__main__":
    # Load or generate new_data_point here
    # Ensure it is a DataFrame with the expected structure
    
    # Example: new_data_point = pd.DataFrame(...)
    
    # Perform real-time classification
    prediction = real_time_classification_SVM(new_data_point, channels, fs)
    
    print("Predicted Label:", prediction)
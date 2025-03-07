import os
import numpy as np
import pandas
import logging
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from EEGNet_Model import EEGNet_SSVEP
from keras.callbacks import Callback
from scipy.signal import butter, sosfilt
from sklearn.model_selection import KFold
from keras.preprocessing.sequence import TimeseriesGenerator


logger = logging.getLogger('Data Import')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

current_dir = os.getcwd()
with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']


def adjust_samples_to_divisible_by_n(samples, labels, n):
    total_samples = samples.shape[0]
    remainder = total_samples % n

    if remainder != 0:
        # If remainder is not 0, adjust the size of samples and labels
        samples = samples[:-remainder]
        labels = labels[:-remainder]

    return samples, labels


def read_data(data_dir):
    # filtering required columns
    eeg_keys = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', "FC1", "C3", "FC5", "FT9", "T7", "CP5", "CP1", "P3", "P7",
                'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4',
                'FC2', 'F4', 'F8', 'Fp2', 'New Label']

    marker_label = ['New Label']
    n = 32

    left_data = []
    right_data = []
    stop_data = []
    left_label = []
    right_label = []
    stop_label = []

    # Initialize label encoder
    label_encoder = LabelEncoder()

    for file in os.listdir(data_dir):
        if file.startswith("eeg"):
            print(f" Loading from: {file}")

            eeg_data = pd.read_csv(os.path.join(data_dir, file)).loc[:, eeg_keys]
            label_df = pd.read_csv(os.path.join(data_dir, file)).loc[:, marker_label]

            eeg_data = eeg_data[eeg_data['New Label'] != 'Rest']
            eeg_data = eeg_data.drop('New Label', axis=1)
            label_df = label_df[label_df['New Label'] != 'Rest']


            # Iterate through each row in label_df
            for index, row in label_df.iterrows():
                label = row['New Label']

                # Get the corresponding EEG data for the current label
                eeg_data_for_label = eeg_data[eeg_data.index == index]
                labels_list = label_df[label_df.index == index]

                # Append the EEG data and label to the respective lists based on the label value
                if label == 'Left':
                    left_data.append(eeg_data_for_label)
                    left_label.append(labels_list)
                elif label == 'Right':
                    right_data.append(eeg_data_for_label)
                    right_label.append(labels_list)
                elif label == 'Stop':
                    stop_data.append(eeg_data_for_label)
                    stop_label.append(labels_list)

    # Convert lists to numpy arrays
    left_data = pd.concat(left_data)
    right_data = pd.concat(right_data)
    stop_data = pd.concat(stop_data)

    left_label = pd.concat(left_label)
    right_label = pd.concat(right_label)
    stop_label = pd.concat(stop_label)

    if left_data.shape[0] % n != 0:
    # If not divisible by 32, adjust the samples and labels
        left_data, left_label = adjust_samples_to_divisible_by_n(left_data, left_label, n)

    if right_data.shape[0] % n != 0:
    # If not divisible by 32, adjust the samples and labels
        right_data, right_label = adjust_samples_to_divisible_by_n(right_data, right_label, n)

    if stop_data.shape[0] % n != 0:
    # If not divisible by 32, adjust the samples and labels
        stop_data, stop_label = adjust_samples_to_divisible_by_n(stop_data, stop_label, n)

    # Convert DataFrames to numpy arrays before reshaping
    left_data = left_data.values
    right_data = right_data.values
    stop_data = stop_data.values

    # Reshape the data for CNN input
    left_data = left_data.reshape(-1, left_data.shape[1], 32, 1)
    right_data = right_data.reshape(-1, right_data.shape[1], 32, 1)
    stop_data = stop_data.reshape(-1, stop_data.shape[1], 32, 1)

    left_label = left_label[:left_data.shape[0]]
    right_label = right_label[:right_data.shape[0]]
    stop_label = stop_label[:stop_data.shape[0]]

    # Combine the data from different classes
    combined_data = np.concatenate((left_data, right_data, stop_data), axis=0)
    # Combine the labels from different classes
    combined_labels = np.concatenate((left_label, right_label, stop_label), axis=0)
    # Flatten the combined_labels array to make it 1-dimensional
    combined_labels_flat = combined_labels.ravel()

    print(f"combined data: {combined_data.shape}, combined labels: {combined_labels.shape}")

    # Convert string labels to numerical labels using LabelEncoder
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(combined_labels_flat)

    # Convert the numerical labels to one-hot encoding
    num_classes = 3
    combined_labels_one_hot = to_categorical(numerical_labels, num_classes)
    print(f"one hot encoded labels: {combined_labels_one_hot.shape}")

    # Split the combined data and labels into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(combined_data, combined_labels_one_hot, test_size=0.1,
                                                      random_state=42, shuffle=True)

    return X_train, X_val, y_train, y_val, num_classes



def train_model():

    X_train, X_val, y_train, y_val, num_classes = read_data(data_dir)

    chans = X_train.shape[1]
    samples = X_train.shape[2]

    model = EEGNet_SSVEP(nb_classes = 3, Chans = chans, Samples = samples,
                   dropoutRate = 0.3, kernLength=256, F1 = 96, D = 1, F2 = 96,
                   dropoutType='Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # count number of parameters in the model
    numParams = model.count_params()
    print(f" number of parameters: {numParams}")

    # set a valid data_dir for your system to record model checkpoints
    # checkpointer = ModelCheckpoint(filedata_dir='/tmp/checkpoint.h5', verbose=1,
    #                                save_best_only=True)
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    # class_weights = {0:1, 1:1, 2:1}

    k = 7  # Number of folds for cross-validation
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize performance metrics lists
    accuracies = []
    losses = []
    models = []

    # Perform time series cross-validation
    for train_index, val_index in cv.split(X_train):
        X_traincv, X_valcv = X_train[train_index], X_train[val_index]
        y_traincv, y_valcv = y_train[train_index], y_train[val_index]

        print(f"X train : {X_traincv.shape}, Y train: {y_traincv.shape}")
        print(f"X val : {X_valcv.shape}, y val: {y_valcv.shape}")

        # Train the model on the training data
        model.fit(X_traincv, y_traincv, epochs=100, batch_size=32, verbose=1)

        # Evaluate the model on the validation data
        loss, accuracy = model.evaluate(X_valcv, y_valcv, verbose=1)

        print(f"Validation loss: {loss}, validation acc: {accuracy}")

        # Record the performance metrics
        accuracies.append(accuracy)
        losses.append(loss)
        models.append(model)

    # Calculate the average performance metrics over all folds
    avg_accuracy = np.mean(accuracies)
    avg_loss = np.mean(losses)

    # Identify the best model based on the highest accuracy
    best_fold_index = np.argmax(accuracies)
    best_model = model  # Or create a copy of the model if needed

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    # Load the best model weights based on the highest accuracy
    best_model.save('checkpoint.h5')

    return best_model, X_val, y_val


def evaluate_model():
    best_model, X_val, y_val = train_model()

    # Evaluate the best model on the validation data
    loss, accuracy = best_model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation loss for the best model: {loss}, validation acc: {accuracy}")


#tarin_model() for validation + training
#evaluate_model() is for
train_model()

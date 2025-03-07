import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from RobotControl import RobotControl

robot_controller = RobotControl()

eeg_keys = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', "FC1", "C3", "FC5", "FT9", "T7", "CP5", "CP1", "P3", "P7",
            'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4',
            'FC2', 'F4', 'F8', 'Fp2']

classification_map = {
    0: 'left',
    1: 'right',
    2: 'stop',
    3: 'forward'
}

# def classify_realtime_chrononet(model_path='best_model_checkpoint.h5', expected_features=32, segment_length=32):
#     """
#     Classify real-time EEG data using a pre-trained ChronoNet model.
#     """
#     path = os.path.join(os.getcwd(), 'profile', 'hybrid_data')
#     concatenated_data = pd.DataFrame()
#
#     for file in os.listdir(path):
#         if file.endswith(".csv"):
#             print(f" Loading from: {file}")
#             file_data = pd.read_csv(os.path.join(path, file)).loc[:, eeg_keys]
#             concatenated_data = pd.concat([concatenated_data, file_data], ignore_index=True)
#
#     new_eeg_data = concatenated_data.values
#     total_length = new_eeg_data.shape[0]
#     num_segments = total_length // segment_length
#
#     if num_segments == 0:
#         raise ValueError(
#             f"Data length {total_length} is less than segment_length {segment_length}. Need more data to make predictions.")
#
#     new_eeg_data = new_eeg_data[:num_segments * segment_length].reshape(
#         (num_segments, segment_length, expected_features))
#
#     # Load the pre-trained ChronoNet model
#     model = load_model(model_path)
#
#     # Predict the class label for each segment
#     for i in range(num_segments):
#         segment = new_eeg_data[i:i + 1]  # Select the current segment with shape (1, segment_length, expected_features)
#         predictions = model.predict(segment)
#         class_label = np.argmax(predictions, axis=1)[0]
#         predicted_action = classification_map[class_label]
#         print(f"Segment {i + 1}, Class Label: {class_label}, Predicted Action: {predicted_action}")
#
#     return

def classify_realtime_EEGNET(new_eeg_data, model_path='profile\\combo data\\EEGNet_4C_bestModel.h5', expected_features=32,
                                segment_length=32):
    """
    Classify real-time EEG data using a pre-trained EEGNet model.
    """

    new_eeg_data = new_eeg_data.loc[:, eeg_keys]

    if isinstance(new_eeg_data, pd.DataFrame):
        new_eeg_data = new_eeg_data.values

    # Reshape the segmented EEG data to have shape (pointwise=1,Channels=32, Times_stamps=32,1).
    new_eeg_data = new_eeg_data[:new_eeg_data * segment_length].reshape(
        (new_eeg_data, segment_length, expected_features))

    print(f"Shape of segmented data after reshaping: {new_eeg_data}")

    # Load the RNN GRU model
    model = load_model(model_path)
    # Predict the class label for the segment using the pre-trained ChronoNet model.
    predictions = model.predict(new_eeg_data)
    class_label = np.argmax(predictions, axis=1)[0]
    predicted_action = classification_map[class_label]
    print(f"Class Label: {class_label}, Predicted Action: {predicted_action}")
    robot_controller.command_robot(predicted_action)

    return

# path = os.path.join(os.getcwd(), 'profile', 'hybrid_data')
# concatenated_data = pd.DataFrame()
#
# for file in os.listdir(path):
#     if file.endswith(".csv"):
#         print(f" Loading from: {file}")
#         file_data = pd.read_csv(os.path.join(path, file)).loc[:, eeg_keys]
#         new_eeg_data = file_data

# <<<<<<< HEAD
# # Call the function
# #classify_realtime_chrononet(new_eeg_data)
# =======
#
# >>>>>>> fd9f84edfc919ed327a07a24cfee664a23557e4d

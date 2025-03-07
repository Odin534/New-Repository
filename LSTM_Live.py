import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication
from RobotControl import RobotControl

def live_classification(new_data_point, model_path):
    """
    Function for live classification of a new EEG data point.
    
    Parameters:
        - new_data_point: New EEG data, should be a pandas DataFrame.
        - model_path: Path to the saved LSTM model.
        
    Returns:
        - predicted_label: The predicted label for the new data point.
    """
    # Load the trained LSTM model
    model = load_model(model_path)
    
    # Ensure the data is 3D: (num_samples, 1, num_features)
    # Assuming new_data_point is a DataFrame with one row per time step, and one column per feature
    new_data_point = np.reshape(new_data_point.values, (1, new_data_point.shape[0], new_data_point.shape[1]))
    
    # Make the prediction
    prediction = model.predict(new_data_point)
    
    # Translate numeric predictions back to class labels if needed
    label_mapping = {
        0: 'stop',
        1: 'left',
        2: 'right',
        3: 'forward'
    }
    
    # Retrieve the index of the maximum value
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    
    # Map the prediction to the corresponding label
    predicted_label = label_mapping[predicted_label_index]
    
    return predicted_label

# Example usage:
if __name__ == "__main__":
    app = QApplication([])
    
    # Load or generate new_data_point here
    # Ensure it is a DataFrame with the expected structure
    # Example: new_data_point = pd.DataFrame(...)
    
    # Specify the path to your saved model
    model_path = 'path_to_your_saved_model.h5'  
    
    # Perform live classification
    prediction = live_classification(new_data_point, model_path)
    
    # Instantiate robot control and send command
    robot_controller = RobotControl()
    robot_controller.command_robot(prediction)

import os, json
import pandas as pd
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier

current_dir = os.getcwd()
with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']

def load_data(data_dir):
    dfs = []
    print("Loading data from directory:", data_dir)
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            print(f"Loading file: {filename}")
            df = pd.read_csv(os.path.join(data_dir, filename))

            # Remove unwanted columns
            df = df.drop(columns=["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag",
                                  "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time"])
            dfs.append(df)

    # Concatenate all dataframes
    df_all = pd.concat(dfs)
    print(f"Concatenated all files. Total samples: {len(df_all)}")
    return df_all

def apply_csp_sgd(data):
    from sklearn.preprocessing import StandardScaler

    # Filter the data for non-'Rest' labels
    df_filtered = data[data["New Label"] != "Rest"]

    # Define features and labels
    X = df_filtered.drop("New Label", axis=1).values
    y = df_filtered["New Label"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Reshape input to be 3D [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Split the dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the CSP transformer
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    print("CSP transformer created.")

    # Apply CSP on the train set
    X_train_csp = csp.fit_transform(X_train, y_train)
    print("CSP applied to the training set.")

    # Apply CSP on the test set
    X_test_csp = csp.transform(X_test)
    print("CSP applied to the test set.")

    # Data normalization
    scaler = StandardScaler()
    X_train_csp = scaler.fit_transform(X_train_csp)
    X_test_csp = scaler.transform(X_test_csp)

    # Define the SGD model
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.006, learning_rate='optimal', eta0=0.01, early_stopping=False, validation_fraction=0.3, n_iter_no_change=5, random_state=42)
    #sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.007, learning_rate='constant', eta0=0.01, early_stopping=False, validation_fraction=0.3, n_iter_no_change=5, random_state=42)

    # Fit the model
    sgd.fit(X_train_csp, y_train)
    print("SGD model trained.")

    # Get predictions
    y_pred = sgd.predict(X_test_csp)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=le.classes_)

    print("Model accuracy: ", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    
if __name__ == "__main__":
    data_dir = os.path.join(current_dir, 'Profile', current_profile)
    data = load_data(data_dir)
    apply_csp_sgd(data)

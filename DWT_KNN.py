import math

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import FastICA
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import pandas as pd
import os, json
import pywt
from sklearn.metrics import confusion_matrix

segment_size = 256
current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    # profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']


label_dict = {0:'left', 1:'right', 2:'stop' }

def preprocess_and_train(data_dir):
    dfs = []
    print("Preprocessing data from directory:", data_dir)
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            print(f"Processing file: {filename}")

            # Load the data
            data = pd.read_csv(os.path.join(data_dir, filename))
            dfs.append(data)

    # Concatenate all the dataframes into one
    all_data = pd.concat(dfs, ignore_index=True)

    #Removing data with label Rest
    data_non_rest = all_data[all_data['New Label'] != 'Rest']
    data_non_rest = data_non_rest[data_non_rest['New Label'] != 'Forward']

    # Select EEG columns based on the 10-20 system
    eeg_columns = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5',
                   'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4',
                   'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

    #Extracting only EEG channel data for further processing
    eeg_data = data_non_rest[eeg_columns]

    # Print the shape of eeg_data for debugging
    print(f"eeg_data shape: {eeg_data.shape}")
    print("ICA followed by DWT is performed to extract features. PCA used to reduce dimensions of said features")

    #Independent Component analysis for Preprocessing(Artifact removal)
    ica = FastICA(whiten='unit-variance')
    eeg_data_ica = pd.DataFrame(ica.fit_transform(eeg_data), columns=eeg_columns) #can we add the label column to the ica data?
    eeg_data_ica_with_label = pd.concat((eeg_data_ica,data_non_rest['New Label']), axis= 1, join= 'inner')

    #Implementation of Discreete wavelenght Transform for Feature Extraction process.
    def dwt(input_data):

        #datastructure for Feature vectors
        feature_vector_list = []
        feature_vector = np.zeros(5)

        # segmenting data into 256 sample chunks
        i = 0
        while (i < len(input_data)):
            segment = input_data[i:i + segment_size]
            i = i + segment_size

            #Decomposing the signal into subbands and getting the coefficients
            coeffs = pywt.wavedec(segment, 'db4', level=4)

            #Calculating the energy of the subbands from the detailed coefficients
            for c in range(len(coeffs)):
                E_subband = np.sum(np.absolute(coeffs[c]) ** 2)
                feature_vector[c] = E_subband
            feature_vector_list.append(feature_vector)

        #Returns feature vector list
        return feature_vector_list

    #Function to create the Feature Vector set with labels
    def create_feature_label_vector(input_data):

        eeg_data_ica_with_label = input_data

        #Segregating data according to labels
        data_left = eeg_data_ica_with_label[eeg_data_ica_with_label['New Label'] == 'Left']
        data_right = eeg_data_ica_with_label[eeg_data_ica_with_label['New Label'] == 'Right']
        data_stop = eeg_data_ica_with_label[eeg_data_ica_with_label['New Label'] == 'Stop']

        #Calculating the number of segements(viz-a-viz feature vectors) belonging to each label
        n_segments_left = math.ceil(data_left.shape[0]/segment_size)
        n_segments_right = math.ceil(data_right.shape[0]/segment_size)
        n_segments_stop = math.ceil(data_stop.shape[0]/segment_size)

        #Total number of segemts(viz-a-viz feature vectors) from the provided dataset
        n_segments = n_segments_left+n_segments_right+n_segments_stop

        #Datastructure to store the feature vectors of different classes
        feature_vector_left = []
        feature_vector_right = []
        feature_vector_stop = []

        for channel in eeg_columns:
            np.array(feature_vector_left.append(dwt(data_left[channel])))
            np.array(feature_vector_right.append(dwt(data_right[channel])))
            np.array(feature_vector_stop.append(dwt(data_stop[channel])))

        #Joining the 3 feature sets to make the complete feature set in the order of the parameters passed (3-d array)
        feature_vector_matrix = np.concatenate((feature_vector_left, feature_vector_right, feature_vector_stop), axis=1)

        #3-d array cannot be passed as parameter to our classifier.Hence we change it to a 2-d array with same order)
        feature_vector_set = np.zeros((feature_vector_matrix.shape[0] * feature_vector_matrix.shape[1],
                                         feature_vector_matrix.shape[2]))
        #Populating feature_vector_set
        for i in range(feature_vector_matrix.shape[0]):
            feature_vector_set[i * n_segments: (i + 1) * n_segments, :] = feature_vector_matrix[i, :, :]

        print(f"feature_vector_set shape:",{feature_vector_set.shape})

        #Returns feature_vector_set (2-d array) and corresponding label vector(1-d array)
        return feature_vector_set,np.hstack((np.zeros(n_segments_left*32),
                                               np.ones(n_segments_right*32),
                                               np.ones(n_segments_stop*32)*2))

    #Create the Feature Vector set
    feature_vector_set, label = create_feature_label_vector(eeg_data_ica_with_label)

    # Apply Principal Component Analysis for dimensionality reduction on the Feature Set
    n_components = min(10, feature_vector_set.shape[1])
    pca = PCA(n_components=n_components)
    feature_vectors_list_pca = pca.fit_transform(feature_vector_set)

    # Split the Feature Set into training, validation, and test sets
    X, X_test, y, y_test = train_test_split(feature_vectors_list_pca, label , test_size=0.2,
                                            random_state=42)

    # Create a KNN
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5 , p = 2, metric='euclidean'))
    print("ICA for Artifact removal and DWT for feature extraction is completed !")
    print("KNN classifier Training is going on !")

    # Cross-validation on the training set
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print(f"Cross-validation scores: {scores}")
    print(f"Average cross-validation score: {scores.mean()}")

    # Train the classifier on the whole training set
    clf.fit(X, y)

    # Evaluate the classifier on the held-out test set
    y_test_pred = clf.predict(X_test)

    # Print classification report
    print("Test set:")
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    #print('The Confusion Matrix is :\n', cm)
    print('The Confusion Matrix is :\n',cm)
    #Normalized Confusion matrix
    cm_nrm = cm/np.sum(cm,axis=1).reshape(-1,1)
    print('The normalized Confusion Matrix is :\n',cm_nrm)


# Call the function
preprocess_and_train(data_dir)
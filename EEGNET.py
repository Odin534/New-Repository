import json
import time
import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from matplotlib import  pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import SpatialDropout2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D,SeparableConv2D
from tensorflow.keras.layers import Input, Activation,AveragePooling2D,Flatten,Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

## this model is doing the forward node in which it is defined the sequence of the operation done in the classier the next step shoud
### be then how to define in wights(paramters) of the model using the loss function
##################### in this part of the code we only load the data from the last selected profile #########
###############
app = QApplication([])
current_dir = os.getcwd()
with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']

def load_data(data_dir, processed_files):
    dfs = []
    print("Loading data from directory:", data_dir)
    for filename in os.listdir(data_dir):
        if filename.startswith("hybrid"):
            print(f"Loading file: {filename}")
            processed_files.append(filename)

            df = pd.read_csv(os.path.join(data_dir, filename))

            # Remove unwanted columns
            # df = df.drop(columns=["COUNTER", "INTERPOLATED", "HighBitFlex", "SaturationFlag",
            #                       "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "Human Readable Time"])
            df = df[['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9'
                , 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2','New Label']]

            #df = df[['PO9','O1','Oz','O2','PO10','New Label']]
            #df = df[[ 'O1', 'Oz', 'O2', 'New Label']]
            #df = df[['P3', 'P7', 'PO9'
            #         , 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4','New Label']]
            dfs.append(df)

    # Concatenate all dataframes
    try:
        df_all = pd.concat(dfs)
    except ValueError:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Please add new files to analyze")
        msg.setWindowTitle("No Data Error")
        msg.exec_()
        return None
    print(f"Concatenated all files. Total samples: {len(df_all)}")
    return df_all
###################
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
def equal_classes(data_x,data_y):
#This function equqlizes the number  of feature vectors and their  corresponding labels
# the basic idea is to shuffle the indices of each class separatley and select the minimal class number
    L = []
    R = []
    S = []
    #Rest = []
    for i in range(len(data_y)): # detect the indices corresponding to each classes
        if data_y[i] == 0:
            L.append(i)
        if data_y[i] == 1:
            R.append(i)
        if data_y[i] == 2:
            S.append(i)
        # if data_y[i] == 3:
        #     Rest.append(i)
    print('Input data decomposition :',len(L),'/',len(R),'/',len(S)) #,len(Rest))
    dim = np.array([len(L),len(R),len(S)])#,len(Rest)])
    random.shuffle(L) # shuffle Left indices
    random.shuffle(R) # shuffle Right indices
    random.shuffle(S) # shuffle Stop indices
    # random.shuffle(Rest) # shuffle Stop indices
    # select the minimal length
    min_length = np.amin(dim)
    # the following 3 lines selects a total number of indices equal to the minimal
    # class's length because that we had used shuffle previously so our choice of data from 0 to min-1 is not biased
    L =L[0:min_length-1]
    R =R[0:min_length-1]
    S =S[0:min_length-1]
    # Rest =Rest[0:min_length-1]
    all_class_idx =np.concatenate((L,R,S), axis=None) #np.concatenate((L,R,S,Rest), axis=None) # reassemble class indices with equal percentage of presence :)
    df_x = [data_x[i,:,:,:] for i in all_class_idx ] # select from the feature matrix only the remaining indices in variable 'all_class_idx'
    df_y = [data_y[i] for i in all_class_idx ] # select from the label vector only the remaining indices in variable 'all_class_idx'
    df_x = np.array(df_x)
    df_y = np.array(df_y)
    return df_x,df_y
def EEGNet(data):
    ###### this function has aim to reshape the data ane make them consommable by Keras ########

    df=data            #df = data[data["New Label"] != "Rest"] # data
    df = df[df["New Label"] != "Rest"]
    df = df[df["New Label"] != "Forward"]

    d_len = 32
    #df_filtered =df[['PO9','O1','Pz','Oz','O2','PO10']].values
    #df_filtered = df[[ 'O1', 'Oz', 'O2']].values
    #df_filtered = df[['P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4']].values
    df_filtered = df[['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9'
                 , 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']].values
    length,channels = df_filtered.shape

    all_lbl = df["New Label"].values
    le = LabelEncoder()
    all_lbl = le.fit_transform(all_lbl)

    i = 0
    void_list = np.linspace(1,d_len,d_len, dtype=int)
    data_x = [] #np.zeros((1,5,64))
    data_y = [] #0
    while i < length :
        if length - i >= d_len:
            data_set = df_filtered[i:i+d_len,:] # select the data set containing 64 points
            fst_lbl=all_lbl[i] #select the label of the first point
            lst_lbl = all_lbl[i:i+d_len]

            lbl_chg = void_list[lst_lbl != fst_lbl]

            if len(lbl_chg) == 0 : # if the same label is present in the selected data then
                # if i == 0:  # in the first step we should create (1,6,64)
                #     data_x = np.expand_dims(data_set.T, axis=0)
                #     data_y = fst_lbl
                # else:
                data_set = np.expand_dims(data_set.T, axis=0)
                data_x.append(data_set)
                data_y.append(fst_lbl)
                # append the structure with the data

                i += d_len
            else:
                i += np.amin(lbl_chg) # do not append the data structure and move to the point where the label has changed
        else:
            i = length
        #print(i,'/',length)

    #data split
    data_x=np.array(data_x)
    data_x= np.reshape(data_x,(data_x.shape[0],data_x.shape[2],data_x.shape[3],1))
    data_y=np.array(data_y)
    data_x,data_y = equal_classes(data_x,data_y)
    data_y = to_categorical(data_y)

    shapee= data_x.shape
    print('the overall shape is ,', shapee)
    print(shapee[0],shapee[1],shapee[2])
    input_shape = shapee[0]

    ## this part is only for best model loading an then measure the time of excusion of classification of a an input.
    # model2 = load_model('C:\\Users\\DELL XPS 15\\Desktop\\New_Repo_main2\\new-repo-main\\profile\\combo data' /
    #                     '\\Best_results_kfold_4_classes_EEGNET\\combo data_1_model.h5') #load model
    # duration_vec = [] # vector excution time
    # for i in range(input_shape): # loop over all data window
    #     start = time.time() # time of start
    #     testpoint = np.reshape(data_x[i, :, :], (1, 32, 32, 1)) # applied preprocessing(filtering , reshape ...)
    #     model2.predict(testpoint) # predict point
    #     end =time.time() # time of end
    #     duration=end - start # duration
    #     duration_vec.append(duration) # vector containing all point wise duration
    # print('the mean excution time ',np.mean(duration_vec),' and the covariance is',np.cov(duration_vec)) # dispaly the mean and the covariance



    # x_train , x_vt ,y_train , y_vt = train_test_split(data_x,data_y,train_size=0.7,random_state=42,stratify=data_y)
    x_train_val , x_test ,y_train_val , y_test = train_test_split(data_x,data_y,train_size=0.85,random_state=42,stratify=data_y)
    #x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, train_size=0.5, random_state=42,stratify=y_vt)
    #print(y_test)


    ## EEGNET Configuration
    nb_classes = 3
    channels = 32
    nb_samples = 32
    dropoutRate = 0.3
    kernLength = nb_samples
    F1 = 96
    D = 1
    F2 = 96
    dropoutType = 'Dropout'


    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_num = 1

    for fold_num, (train_index, test_index) in enumerate(skf.split(x_train_val, y_train_val.argmax(1))):
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        model_path = os.path.join(data_dir, f'best_model_fold_{fold_num}.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:

            model = Sequential()
            model.add(Conv2D(F1,(1,kernLength), padding = 'same',
                            input_shape = (channels,nb_samples,1),
                            use_bias = False))  # apply the first convolution where no bias is applied)
            model.add(BatchNormalization())
            model.add(DepthwiseConv2D((channels,1),use_bias=False, depth_multiplier= D,
                                    depthwise_constraint = max_norm(1.)))
            model.add(BatchNormalization())
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1,4)))
            model.add(Dropout(dropoutRate))
            model.add(SeparableConv2D(F2,(1,16),use_bias = False, padding= 'same'))
            model.add(BatchNormalization())
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1,8)))
            model.add(Dropout(dropoutRate))
            model.add(Flatten(name = 'flatten'))
            model.add(Dense(nb_classes, name='dense'))
            model.add(Activation('softmax',name='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model Summary:')
        model.summary()
        # Define early stopping and model checkpointing
        #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)



        # Fit the model
        print(f"Started training EEGNET model for fold {fold_num}...")
        history=model.fit(x_train,y_train, epochs = 2, validation_data=(x_val,y_val),
                          callbacks=[checkpoint, TestCallback((x_test, y_test))])
        # Print training and validation loss and accuracy per epoch for this fold
        print(f"Fold {fold_num} - Training Loss per epoch:", history.history['loss'])
        print(f"Fold {fold_num} - Validation Loss per epoch:", history.history['val_loss'])
        print(f"Fold {fold_num} - Training Accuracy per epoch:", history.history['accuracy'])
        print(f"Fold {fold_num} - Validation Accuracy per epoch:", history.history['val_accuracy'])

        # Evaluate the model on the test data for this fold
        _, accuracy = model.evaluate(x_test, y_test)
        print(f"Fold {fold_num} - Test set accuracy: ", accuracy)


        #Setting max values as 1 and other as 0
        y_prediction = model.predict(x_test)
        # y_pred = np.argmax(y_prediction, axis=1,keepdims=True)
        max = y_prediction.max(axis=1).reshape(-1, 1)
        # Setting max values as 1 and other as 0
        y_pred = np.where(y_prediction == max, 1, 0)

        y_testing = np.asarray(y_test, dtype='int')
        cm = confusion_matrix(y_testing.argmax(axis=1), y_pred.argmax(axis=1))
        # print('The Confusion Matrix is :\n', cm)
        print('The Confusion Matrix is :\n')
        print(cm)
        # Normalized Confusion matrix
        cm_nrm = cm / np.sum(cm, axis=1).reshape(-1, 1)
        print(f"Fold {fold_num} - Normalized Confusion Matrix:")
        print(cm_nrm)


        model_path_name = data_dir+"\\"+current_profile+"_"+str(fold_num)+"_model.h5"
        model.save(model_path_name)
        print(f"Fold {fold_num} - Classification Report:")
        print(classification_report(y_testing.argmax(axis=1), y_pred.argmax(axis=1), target_names=le.classes_))



    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.0, 1.0])
    # plt.xticks([0, 10, 20, 30,40,50,60,70,80,90,100,110,120,130,140])
    # plt.legend(loc='lower right')
    # plt.show()


    # input1 = Input(shape = (channels,nb_samples,1))
    #
    # block1 = Conv2D(F1,(1,kernLength), padding = 'same',
    #                 input_shape = (channels,nb_samples,1),
    #                 use_bias = False)(input1) # apply the first convolution where no bias is applied
    # block1 = BatchNormalization()(block1)
    # block1 = DepthwiseConv2D((channels,1),use_bias=False, depth_multiplier= D,
    #                         depthwise_constraint = max_norm(1.))(block1)
    # block1 = BatchNormalization()(block1)
    # block1 = Activation('elu')(block1)
    # block1 = AveragePooling2D((1,4))(block1)
    # block1 = dropoutType(dropoutRate)(block1)
    #
    # block2 = SeparableConv2D(F2,(1,16),use_bias = False, padding= 'same')(block1)
    # block2 = BatchNormalization()(block2)
    # block2 = Activation('elu')(block2)
    # block2 = AveragePooling2D((1,8))(block2)
    # block2 = dropoutType(dropoutRate)(block2)
    # flatten = Flatten(name = 'flatten')(block2)
    # dense = Dense(nb_classes, name='dense')(flatten)
    # softmax = Activation('softmax',name='softmax')(dense)

    #return Model(inputs = input1, ouputs = softmax)
################### this function is used in order to perform the hole optimisation #####

# def apply_eegnet(data):
#     # Filter the data for non-"Rest" labels
#     df_filtered = data[data["New Lable"] != "Rest"]
#
#     # define the features and their labels
#     x = df_filtered.drop("New Label", axis = 1).values
#     y = df_filtered["New Label"].values
#
#     # Encode the labels in a hot vector
#     le = LabelEncoder()
#     y = le.fit_transform(y)
#     y = to_categorical(y)
################################
if __name__ == "__main__":
    data_dir = os.path.join(current_dir, 'Profile', current_profile)
    processed_files_path = f'{data_dir}/processed_files.json'
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = json.load(f)
    else:
        processed_files = []
    data = load_data(data_dir, processed_files)
    if data is not None:
        EEGNet(data)
    else:
        print("No new data to analyze.")
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f)
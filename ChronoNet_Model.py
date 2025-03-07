import logging
from scipy.signal import butter, sosfilt
import numpy as np
import pandas as pd
import keras

from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, GRU, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
class CHRONONET():

    def __init__(self):
        self.logger = logging.getLogger('CHRONONET')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

    def train(self, data, classes):
        Chans = data.shape[2]
        Samples = data.shape[1]


        def block(input):
            conv1 = Conv1D(32, 2, strides=2, activation='relu', padding="same")(input)
            conv2 = Conv1D(32, 4, strides=2, activation='relu', padding="causal")(input)
            conv3 = Conv1D(32, 8, strides=2, activation='relu', padding="causal")(input)
            x = concatenate([conv1, conv2, conv3], axis=2)
            return x

        input = Input(shape=(Samples, Chans))
        print(f"input shape {input.shape}")
        block1 = block(input)
        block2 = block(block1)
        block3 = block(block2)

        gru_out1 = GRU(32, activation='tanh', return_sequences=True)(block3)
        gru_out2 = GRU(32, activation='tanh', return_sequences=True)(gru_out1)
        gru_out = concatenate([gru_out1, gru_out2], axis=2)
        gru_out3 = GRU(32, activation='tanh', return_sequences=True)(gru_out)
        gru_out = concatenate([gru_out1, gru_out2, gru_out3])
        gru_out4 = GRU(32, activation='tanh')(gru_out)

        predictions = Dense(classes, activation='softmax')(gru_out4)
        model = Model(inputs=input, outputs=predictions)

        return model
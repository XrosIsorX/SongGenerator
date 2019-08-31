from NeuralNetwork.NeuralNetwork import NeuralNetwork

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

class CNN_LSTM_Network(NeuralNetwork):
    def __init__(self, model_name="CNN_LSTM"):
        super().__init__(model_name=model_name)
    
    def build_model(self, input_shape, output_shape): # input_shape Ex. (x.shape[1], x.shape[2], 1)
        model = Sequential()

        model.add(Conv1D(50, kernel_size=(1000, ), activation='linear', strides=500, input_shape=input_shape, padding='same'))
        model.add(Conv1D(25, kernel_size=(5,), strides=5, activation='linear', padding='same'))
        model.add(Conv1D(1, kernel_size=(1,),activation='linear', padding='same'))
        model.add(MaxPooling1D(pool_size=(2,), padding='same'))
        # model.add(Flatten())

        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(100))

        model.add(Dense(100))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape[1]))

        model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(),metrics=['mae'])

        model.summary()

        self.model = model

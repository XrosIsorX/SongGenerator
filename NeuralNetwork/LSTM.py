from NeuralNetwork import NeuralNetwork

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

class LSTM_Network(NeuralNetwork):
    def __init__(self, model_name="LSTM"):
        super().__init__(model_name=model_name)
    
    def build_model(self, input_shape, output_shape): # input_shape Ex. (x.shape[1], x.shape[2], 1)
        model = Sequential()
        model.add(LSTM(
            128,
            input_shape=input_shape,
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        # model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.3))
        model.add(LSTM(100))
        # model.add(Dense(100))
        # model.add(Dropout(0.3))
        model.add(Dense(output_shape[1]))

        model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(),metrics=['mae'])

        model.summary()

        self.model = model

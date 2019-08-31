import librosa
import numpy as np
import os
import config 
import matplotlib.pyplot as plt

from NeuralNetwork.CNN_LSTM import CNN_LSTM_Network
import utils.DataProcessor as dp

violin_filenames = os.listdir(config.violin_dir)

violin_path = config.violin_dir + violin_filenames[0]

song, sr = librosa.load(violin_path)

x, y = dp.build_sliding_window(song, x_shape=100000, y_shape=1000)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))
y = np.reshape(y, (y.shape[0], y.shape[1]))

# model = CNN_LSTM_Network()
# model.build_model((x.shape[1], x.shape[2]), y.shape)
model = CNN_LSTM_Network("CNN_LSTM_0100_0200_0100_0300")
model.load_weights((x.shape[1], x.shape[2]), y.shape)

model.train(x, y,epochs=1000, batch_size=32, checkpoint=5)

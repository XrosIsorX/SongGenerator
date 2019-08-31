import librosa
import numpy as np
import os
import config 
import matplotlib.pyplot as plt

from NeuralNetwork.LSTM import LSTM_Network
import utils.DataProcessor as dp

violin_filenames = os.listdir(config.violin_dir)

violin_path = config.violin_dir + violin_filenames[0]

song, sr = librosa.load(violin_path)

x, y = dp.build_sliding_window(song)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))
y = np.reshape(y, (y.shape[0], y.shape[1]))

model = LSTM_Network()
model.build_model((x.shape[1], x.shape[2]), y.shape)
model.train(x, y, batch_size=512, checkpoint=5)

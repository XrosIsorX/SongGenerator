import librosa
import numpy as np
import os
import config 
import matplotlib.pyplot as plt

from LSTM import LSTM_Network

def build_sliding_window(array, x_shape=300, y_shape=100):
    x = []
    y = []
    for i in range(x_shape, len(array) - y_shape, y_shape):
        x.append(array[i - x_shape: i])
        y.append(array[i: i + y_shape])
    return np.array(x), np.array(y)

violin_filenames = os.listdir(config.violin_dir)

violin_path = config.violin_dir + violin_filenames[0]

song, sr = librosa.load(violin_path)

x, y = build_sliding_window(song)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))
y = np.reshape(y, (y.shape[0], y.shape[1]))

model = LSTM_Network()
model.build_model((x.shape[1], x.shape[2]), y.shape)
model.train(x, y, batch_size=512, checkpoint=5)

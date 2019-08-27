import numpy as np
import os
import config 

import librosa

from LSTM import LSTM_Network

def build_random_input():
    x = np.random.rand(1, 300)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x

x = build_random_input()

model = LSTM_Network("LSTM_0020")
model.load_weights((x.shape[1], x.shape[2]), (1, 100))

def generate(sequence=1000, start_song=None):
    if type(start_song) == type(None): 
        x = build_random_input()
    else:
        x = np.array([start_song])
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    song = [v[0] for v in x[0]]
    for i in range(sequence):
        print(i)
        next_sound = model.predict(x)
        song += list(next_sound[0])
        x = list(x[0][100:]) + list(next_sound[0])
        x = np.reshape(x, (1, 300, 1))
    return song

violin_filenames = os.listdir(config.violin_dir)

violin_path = config.violin_dir + violin_filenames[0]

song, sr = librosa.load(violin_path)

song = song[27000:27300]

song = generate(sequence=3000, start_song=song)

a = song[300:]

librosa.output.write_wav('file_trim_5s.wav', np.array(a), 22050)

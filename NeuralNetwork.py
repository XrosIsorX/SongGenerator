import os

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class NeuralNetwork:
    def __init__(self,  model_dir = "models/models/",
                        log_dir = "models/logs/",
                        config_dir = "models/configs/",
                        model_name = ""
                        ):

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.config_dir = config_dir

        self.model_name = model_name

        # Keras setting
        self.csv_logger = CSVLogger(self.log_dir + self.model_name + ".log")

        # Prepare
        self.create_directory()

    def create_directory(self):
        if not os.path.exists(os.path.dirname(self.model_dir)):
            os.makedirs(os.path.dirname(self.model_dir))
        if not os.path.exists(os.path.dirname(self.log_dir)):
            os.makedirs(os.path.dirname(self.log_dir))
        if not os.path.exists(os.path.dirname(self.config_dir)):
            os.makedirs(os.path.dirname(self.config_dir))

    # Override method to build model
    def build_model(self, input_shape, output_shape):
        self.model = None

    def load_weights(self, input_shape, output_shape):
        self.build_model(input_shape, output_shape)
        self.model.load_weights(self.model_dir + self.model_name + ".h5")

    def load_model(self):
        self.model = load_model(self.model_dir + self.model_name + ".h5py")
        print("Load full model : ", self.model_name)
    
    def save_config(self):
        f = open(self.config_dir + self.model_name + ".txt", "a")
        f.write(str(self.model.get_config()))
        f.close()

    def save_weights(self):
        self.model.save_weights(self.model_dir + self.model_name + ".h5")
        self.save_config()
    
    def save_model(self):
        self.model.save(self.model_dir + self.model_name + ".h5py")
        self.save_config()

    def train(self, x, y, valid_x=[], valid_y=[], epochs=100, batch_size=32, checkpoint=5, save_weights_only=True, shuffle=True):

        if save_weights_only:
            file_type = "h5"
        else:
            file_type = "h5py"
        checkpoint_path = self.model_dir + self.model_name + "_{epoch:04d}." + file_type
        self.cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=save_weights_only, period=checkpoint)

        if valid_x != [] or valid_y != []:
            model_result = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_x, valid_y), shuffle=shuffle, callbacks=[self.csv_logger, self.cp_callback])
        else:
            model_result = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=shuffle, callbacks=[self.csv_logger, self.cp_callback])
        
        if save_weights_only:
            self.save_weights()
        else:
            self.save_model()

    def predict(self, state):
        return self.model.predict(state)

import keras
from keras import metrics, regularizers

class MovementPredictor(object):

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        input_shape = [115]
        model = keras.Sequential()
        model.add(keras.layers.Dense(40, activation='relu',
                                     kernel_regularizer=regularizers.l2(0.01),
                                     input_shape=input_shape))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(16, activation='relu'))
        return model

    def load_data(self, input_path):
        pass

    def train(self):
        pass

    def infer(self, feature):
        pass



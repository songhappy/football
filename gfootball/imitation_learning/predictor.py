import keras
from keras import regularizers
import random
from collections import deque
import numpy as np

class MovementPredictor(object):

    def __init__(self, action_size, state_shape):
        self.epsilon_min = 0.01
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model(state_shape)

    def _build_model(self, input_shape):
        model = keras.Sequential()
        model.add(keras.layers.Dense(40, activation='relu',
                                     input_shape=input_shape))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(80, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        return model

    def act(self, state):
        # if np.random.rand() <= self.epsilon_min:
        #     return(random.randrange(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self,  states, labels, batch_size, epoch=10):
        self.model.fit(x=states, y=labels, batch_size=batch_size, epochs=epoch)

    def evaluate(self, states, labels, batch_size=64):
        loss, acc=self.model.evaluate(states, labels, batch_size, verbose=1)
        return loss, acc

    def load(self, path_name):
        self.model.load_weights(path_name)

    def save(self, path_name):
        self.model.save_weights(path_name)
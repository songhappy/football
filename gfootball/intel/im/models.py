import keras
from keras.callbacks import ModelCheckpoint
import random
from collections import deque
import numpy as np

class MovementPredictor(object):

    def __init__(self, action_size, state_shape):
        self.state_shape = state_shape
        self.action_size = action_size
        self.lr = 0.001
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
        optimizer = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return model

    def _build_model1(self, input_shape):
        model = keras.Sequential()
        model.add(keras.layers.Dense(80, activation='relu',
                                     input_shape=input_shape))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(160, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(80, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(40, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        optimizer = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return model

    def act(self, state):
        # if np.random.rand() <= self.epsilon_min:
        #     return(random.randrange(self.action_size))
        policy = self.model.predict(state).flatten()
      #  return np.random.choice(self.action_size, 1, p=policy)
        return np.argmax(policy)

    def train(self,  states, labels, batch_size, epoch=10, validation_split=0.1, model_path="./"):
        shaped = np.zeros([len(labels), self.action_size])
        checkpoint = ModelCheckpoint(model_path+"/best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode="min")
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        history = self.model.fit(x=states, y=shaped,\
                                 batch_size=batch_size, epochs=epoch, validation_split=validation_split,\
                                 callbacks=[checkpoint])
        return history

    def evaluate(self, states, labels, batch_size=64):
        shaped = np.zeros([len(labels), self.action_size])
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        loss, acc=self.model.evaluate(states, shaped, batch_size, verbose=0)
        return loss, acc

    def load(self, path_name):
        self.model.load_weights(path_name)

    def save(self, path_name):
        self.model.save_weights(path_name)

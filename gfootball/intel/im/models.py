import keras
from keras.callbacks import ModelCheckpoint
import random
from collections import deque
import numpy as np
from zoo.pipeline.api.net import Net
from zoo.pipeline.api.keras.layers import *

class MovementPredictorKeras(object):

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
        model.add(keras.layers.Dense(40, activation='relu',
                                     input_shape=input_shape))
        model.add(keras.layers.Dense(80, activation="relu"))
        model.add(keras.layers.Dense(20, activation='relu'))
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
        model_new = self._build_model1(self.state_shape)
        model_new.set_weights(self.model.get_weights())
        self.model = model_new


    def save(self, path_name):
        self.model.save_weights(path_name)


class MovementPredictorZoo(object):

    def __init__(self, action_size, state_shape):
        self.state_shape = state_shape
        self.action_size = action_size
        self.lr = 0.001
        self.model = self._build_model(state_shape)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(40, activation='relu',
                                     input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(80, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        optimizer = Adam(learningrate=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return model

    def act(self, state):
        # if np.random.rand() <= self.epsilon_min:
        #     return(random.randrange(self.action_size))
        policy = self.model.predict(state, distributed=False)
        #print(policy)
        #print(np.argmax(policy))
      #  return np.random.choice(self.action_size, 1, p=policy)
        return np.argmax(policy)

    def train(self,  states, labels, batch_size, epoch=10, validation_split=0.1, model_path="./"):
        shaped = np.zeros([len(labels), self.action_size])
        checkpoint = ModelCheckpoint(model_path+"/best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode="min")
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        history = self.model.fit(x=states, y=shaped,\
                                 batch_size=batch_size, nb_epoch=epoch, validation_split=validation_split)
        return history

    def evaluate(self, states, labels, batch_size=64):
        shaped = np.zeros([len(labels), self.action_size])
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        loss, acc=self.model.evaluate(states, shaped, batch_size)
        return loss, acc

    def load(self, path_name):
        model = Net.load(path_name)
        self.model = model

    def save(self, path_name):
        self.model.saveModel(path_name)

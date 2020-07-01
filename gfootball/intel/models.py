import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, AveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential

def mlp(state_size, action_size, hidden_layers=[256, 256, 48]):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=state_size, activation='relu',
                    kernel_initializer='he_uniform'))
    for h in hidden_layers[1:]:
        model.add(Dense(h, activation='relu',
                        kernel_initializer='he_uniform'))

    model.add(Dense(action_size, activation='linear',
                    kernel_initializer='he_uniform'))
    return model


def jiaoda_cnn(state_shape, action_size):
    model = Sequential()
    e_conv1 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', data_format="channels_last", activation=None,
                     input_shape=state_shape)
    e_pool1 = AveragePooling2D(pool_size=2)
    e_conv2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', data_format="channels_last", activation=None)
    e_flat = Flatten()
    e_dense = Dense(64, activation='relu')
    e_out = Dense(action_size, activation='linear')

    model.add(e_conv1)
    model.add(e_pool1)
    model.add(e_conv2)
    model.add(e_flat)
    model.add(e_dense)
    model.add(e_out)
    return model


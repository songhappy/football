import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Dropout
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
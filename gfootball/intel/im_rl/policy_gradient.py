import keras
from keras import regularizers
import random
from collections import deque
import numpy as np

class MovementPredictor(object):

    def __init__(self, action_size, state_shape):
        self.epsilon_min = 0.01
        self.gama = 0.99
        self.maxLength = 100
        self.states = []
        self.actions = []
        self.rewards = []
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

    def act(self, state):
        # if np.random.rand() <= self.epsilon_min:
        #     return(random.randrange(self.action_size))
        policy = self.model.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
        #return np.argmax(policy)

    def memorize(self, state, action, reward, done):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)

    def train(self,  states, labels, batch_size, epoch=10):
        shaped = np.zeros([len(labels), self.action_size])
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        self.model.fit(x=states, y=shaped, batch_size=batch_size, epochs=epoch, verbose=0)

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

    def train_rl(self):
        episode_length = len(self.rewards[-self.maxLength:])
        discounted_reward = self.discount_rewards(self.rewards[-self.maxLength:])
        # print(self.rewards)
        # discounted_reward -= np.mean(discounted_reward)
        # discounted_reward /= np.std(discounted_reward)
        # print(discounted_reward)
        update_inputs = np.zeros([episode_length, self.state_shape[0]])
        advantages = np.zeros([episode_length, self.action_size])

        for i in range(episode_length):
            states = self.states[-self.maxLength:]
            update_inputs[i] = states[i]
            actions = self.actions[-self.maxLength:]
            advantages[i][actions[i]] = discounted_reward[i]

        self.model.fit(update_inputs, advantages, batch_size=self.maxLength, epochs=1, verbose=0)
        self.states = []
        self.actions = []
        self.rewards = []

    def discount_rewards(self, rewards):
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gama + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

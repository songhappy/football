import keras
from keras import regularizers
import random
from collections import deque
import numpy as np

class MovementPredictor(object):

    def __init__(self, action_size, state_shape):
        self.epsilon_min = 0.01
        self.gama = 0.9
        self.state_shape = state_shape
        self.action_size = action_size
        self.maxLength = 2000
        self.states = deque(maxlen=self.maxLength)
        self.actions = deque(maxlen=self.maxLength)
        self.rewards = deque(maxlen=self.maxLength)
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
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        return model

    def act(self, state):
        # if np.random.rand() <= self.epsilon_min:
        #     return(random.randrange(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def memorize(self, state, action, reward, done):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)

    def train(self,  states, labels, batch_size, epoch=10):
        shaped = np.zeros([len(labels), self.action_size])
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        self.model.fit(x=states, y=shaped, batch_size=batch_size, epochs=epoch)

    def evaluate(self, states, labels, batch_size=64):
        shaped = np.zeros([len(labels), self.action_size])
        for i in range(len(labels)):
            shaped[i][int(labels[i])] = 1
        loss, acc=self.model.evaluate(states, shaped, batch_size, verbose=1)
        return loss, acc

    def load(self, path_name):
        self.model.load_weights(path_name)

    def save(self, path_name):
        self.model.save_weights(path_name)

    def train_rl(self):
        episode_length = len(self.rewards)
        discounted_reward = self.discount_rewards(self.rewards)
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        update_inputs = np.zeros([episode_length, self.state_shape[0]])
        advantages = np.zeros([episode_length, self.action_size])

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_reward[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=1)
        self.states = deque(maxlen=self.maxLength)
        self.actions = deque(maxlen=self.maxLength)
        self.rewards = deque(maxlen=self.maxLength)

    def discount_rewards(self, rewards):
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gama + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

from keras import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras import regularizers
import random
from collections import deque
import numpy as np

class PGAgent(object):

    def __init__(self, action_size, state_shape, model_path):
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
        self.model_path = model_path

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(40, activation='relu',
                                     input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(80, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        optimizer = Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return model

    def get_action(self, state):
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
        loss, acc=self.model.evaluate(states, shaped, batch_size, verbose=1)
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

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.states = []
        self.actions = []
        self.rewards = []

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()


    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dropout(0.1))
        actor.add(Dense(52, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dropout(0.1))
        actor.add(Dense(52, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dropout(0.05))
        actor.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        critic.add(Dropout(0.1))
        critic.add(Dense(52, activation='relu',
                        kernel_initializer='he_uniform'))
        critic.add(Dropout(0.1))
        critic.add(Dense(52, activation='relu',
                        kernel_initializer='he_uniform'))
        critic.add(Dropout(0.05))
        critic.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gama + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network every episode
    def train_model(self, states, actions, rewards, done):
        discounted_rewards = self.discount_rewards(rewards)
        state_values = self.critic.predict(np.array(states))
        episode_length = len(states)

        advantages1d = discounted_rewards - np.reshape(state_values, len(state_values))
        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = states[i]
            advantages[i][actions[i]] = advantages1d[i]

        self.actor.fit(np.array(states), advantages, epochs=1, verbose=0)
        self.critic.fit(np.array(states), discounted_rewards, epochs=1, verbose=0)

class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.9999
        self.learning_rate = 0.00008
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.batch_size = 256
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_ddqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(52, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(52, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dropout(0.05))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
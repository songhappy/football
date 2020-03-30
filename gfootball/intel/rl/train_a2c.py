from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import gym

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000


from gfootball.intel.im.preprocess import *

import numpy as np

TRAINPG = True
DURATION = 600
RENDER = False
DUMP = False

model_path = "/home/arda/intelWork/projects/googleFootball/run_keeper/dumpFeb4_model_rl/"

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

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights(model_path + "/actor.h5")
            self.critic.load_weights(model_path+"/critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


def main():    # In case of CartPole-v1, maximum length of episode is 500
    players = ['agent:left_players=1']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': DUMP,
        'players': players,
        'real_time': True,
        'level': 'academy_run_to_score_with_keeper',
    })

    action_size = len(action_set_dict["default"])
    feature_size = 83
    state_size = feature_size

    # make A2C agent
    agent = A2CAgent(state_size, action_size)
    env = football_env.FootballEnv(cfg)
    obs, reward, done, info = env.step(env.action_space.sample())

    scores, episodes = [], []

    nepisode= 0
    while nepisode < 100000:
        score = 0
        done = False
        state = observation_sim(obs)
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = observation_sim(next_state)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                print("episode:{}".format(nepisode), "score:{}".format(score))
                env.reset()
                if reward > 0:
                    reward = np.array([1], dtype=float)
                elif reward == 0:
                    reward = np.array([-0.5], dtype=float)
                else:
                    reward = np.array([-1], dtype=float)
                nepisode = nepisode + 1
                if (score > 0):
                    win = win + 1
                elif score < 0:
                    lose = lose + 1
                score = 0


        # save the model
        if nepisode % 50 == 0:
            agent.actor.save_weights(model_path+"/actor.h5")
            agent.critic.save_weights(model_path+"/critic.h5")
if __name__ == '__main__':
    main()
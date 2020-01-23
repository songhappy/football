from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six.moves.cPickle
from gfootball.env.football_action_set import *
from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers

import keras
from keras import regularizers
import numpy as np
import random
from collections import deque
import time

TRAIN = True
DURATION = 600

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
                                     kernel_regularizer=regularizers.l2(0.01),
                                     input_shape=input_shape))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(self.action_size, activation='relu'))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon_min:
            return(random.randrange(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self,  states, labels, batch_size, epoch=10):
        self.model.fit(x=states, y=labels, batch_size=batch_size, epochs=epoch)

    def load(self, path_name):
        self.model.load_weights(path_name)

    def save(self, path_name):
        self.model.save_weights(path_name)

def observation_sim(obss):
    # except right_agent_sticky_actions, right_agent_controlled_player, score, ball_owner_player, steps_left
    # all other keys are extracted
    final_obs = []
    for obs in obss:
        o = []
        o.extend(obs['ball'])
        o.extend(obs['ball_direction'])
        o.extend(obs['ball_rotation'])
        o.extend(obs['left_team'].flatten())
        o.extend(obs['left_team_direction'].flatten())
        o.extend(obs['left_team_tired_factor'].flatten())
        o.extend(list(map(lambda x: 1 if x else 0, obs['left_team_active'])))
        o.extend(list(map(lambda x: 1 if x else 0, obs['left_team_yellow_card'])))
        o.extend(obs['left_team_roles'].flatten())
        o.extend(obs['right_team'].flatten())
        o.extend(obs['right_team_direction'].flatten())
        o.extend(obs['right_team_tired_factor'].flatten())
        o.extend(list(map(lambda x: 1 if x else 0, obs['right_team_active'])))
        o.extend(list(map(lambda x: 1 if x else 0, obs['right_team_yellow_card'])))
        o.extend(obs['right_team_roles'].flatten())
      #  o.extend(obs['left_agent_sticky_actions'][0].flatten())
     #   o.extend(obs['left_agent_controlled_player'])
        if obs['ball_owned_team'] == -1:
            o.extend([1, 0, 0])
        if obs['ball_owned_team'] == 0:
            o.extend([0, 1, 0])
        if obs['ball_owned_team'] == 1:
            o.extend([0, 0, 1])
        game_mode = [0] * 7
        game_mode[obs['game_mode']] = 1
        o.extend(game_mode)
        final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)

def load_data(dump_file):
    # labels, array(), states, array(n, 70)
    with open(dump_file, 'rb') as f:
        dumps = six.moves.cPickle.load(f)
    print(dumps[0])
    labels=[]
    observations = []
    actions = action_set_dict['default']
    actions_dict = {actions[i]: i for i in range(0, len(actions))}

    for dump in dumps:
        key = dump['debug']['action'][0]
        labels.append(actions_dict[key])
        observations.append(dump['observation'])
    states = observation_sim(observations)
    return(np.array(labels, dtype=np.float32), np.array(states, dtype=np.float32))


def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/dumps/episode_done_20191220-101211706616.dump"
    import os
    print(os.listdir("/home/arda/intelWork/projects/googleFootball/dumps/"))

    players = ['agent:left_players=1']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': True,
        'players': players,
        'real_time': True,
        'level': 'academy_pass_and_shoot_with_keeper',
    })

    actions_size = len(action_set_dict["default"])
    labels, states = load_data(input_path)

    if (not TRAIN):
        agent = MovementPredictor(actions_size, [states[0].size])
        agent.train(states, labels, 2, epoch=10)
    else:
        env = football_env.FootballEnv(cfg)
        env.reset()
        agent = MovementPredictor(actions_size, [states[0].size])
        obs, reward, done, info = env.step(env.action_space.sample())
        begin = time.time()
        end = begin + DURATION
        total_reward = 0
        done = False
        total_checkpoint_reward = 0
        print("________________________")
        print(obs)
        i = 0
        while end > time.time():
            print("step: ", i)
            state = observation_sim(obs)
            action = agent.act(state)
            obs, reward, done, info = env.step([action])
            wrapper = wrappers.CheckpointRewardWrapper(env)
            #checkpoint_reward = wrapper.reward(reward)
            total_reward = total_reward + reward
            #total_checkpoint_reward = total_checkpoint_reward + checkpoint_reward
            i = i + 1
            if done:
                env.reset()
            print("total_reward: ", total_reward)
        #print("total_checkpoint_reward: "+total_checkpoint_reward)

if __name__ == '__main__':
    main()
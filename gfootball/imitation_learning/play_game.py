from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six.moves.cPickle
from gfootball.env.football_action_set import *
from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers
import  sys
import keras
from keras import regularizers
import numpy as np
import random
from collections import deque
import time
import os


TRAIN = True
DURATION = 600
RENDER = False
model_path = "/home/arda/intelWork/projects/googleFootball/dumpFeb4_model"

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
        if np.random.rand() <= self.epsilon_min:
            return(random.randrange(self.action_size))
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

def load_1file(dump_file):
    # labels, array(), states, array(n, 70)
    with open(dump_file, 'rb') as f:
        dumps = six.moves.cPickle.load(f)
    labels=[]
    observations = []
    actions = action_set_dict['default']
    actions_dict = {actions[i]: i for i in range(0, len(actions))}

    for dump in dumps:
        key = dump['debug']['action'][0]
        labels.append(actions_dict[key])
        observations.append(dump['observation'])
        if 'frame' in dump['observation']:
            print(dump['observation']['frame'])
    states = observation_sim(observations)
    return(np.array(labels, dtype=np.float32), np.array(states, dtype=np.float32))

def load_data(input_path):
    files = os.listdir(input_path)
    labels = []
    states = []
    for ifile in files:
        if "_1_" in ifile:
            ilabels, istates = load_1file(input_path + ifile)
            labels.append(ilabels)
            states.append(istates)

    labels = np.concatenate(labels)#.flatten()
    states = np.concatenate(states)#.reshape(351989, 59)
    print(labels.shape)
    print(states.shape)
    return labels, states

def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/dumpFeb4/"

    players = ['agent:left_players=1']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': True,
        'players': players,
        'real_time': True,
        'level': 'academy_run_to_score_with_keeper',
    })

    actions_size = len(action_set_dict["default"])
    feature_size = 83
    if ( not TRAIN):
        labels, states = load_data(input_path)
        train_split = 1700000
        val_split=10000
        train_labels = labels[0:train_split]; val_labels = labels[-val_split:-1]
        train_states= states[0:train_split]; val_states = states[-val_split:-1]

        agent = MovementPredictor(actions_size, [feature_size])
        for nepoch in range(1, 21):
            agent.train(train_states, train_labels, 64, epoch=1)
            loss, acc = agent.evaluate(val_states, val_labels)
            print("epoch={}".format(nepoch), "loss:{}".format(loss), "acc:{}".format(acc))
        agent.save(model_path)
    else:
        env = football_env.FootballEnv(cfg)
        if RENDER:
            env.render()
        env.reset()
        agent = MovementPredictor(actions_size, [feature_size])
        agent.load(model_path)
        #print(agent.model.get_weights())
        obs, reward, done, info = env.step(env.action_space.sample())
        nepisode = 0
        episode_reward = 0
        win = 0
        lose = 0
        while nepisode < 100:
            feature = observation_sim(obs)
            action = agent.act(feature)
            #action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward = episode_reward + reward
            if done:
                env.reset()
                nepisode = nepisode + 1
                if(episode_reward >0): win = win + 1
                elif episode_reward <0: lose = lose + 1

                print("episode:{}".format(nepisode), "episode_reward:{}".format(episode_reward))
                episode_reward = 0
        print("win:{}".format(win), "lose:{}".format(lose))

if __name__ == '__main__':
    main()
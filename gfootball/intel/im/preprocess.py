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
score_length = 100

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

def filter_positives(labels, states, rewards, score_length):
    indices = []
    labels_out = []
    states_out = []
    rewards_out = []
    for i in range(len(rewards)):
        if rewards[i] == 1:
            indices.append(i)
    #print(indices)
    for i in indices:
        labels_out.extend(labels[i-score_length+1:i+1])
        states_out.extend(states[i-score_length+1:i+1])
        rewards_out.extend(rewards[i-score_length+1:i+1])

    return labels_out, states_out, rewards_out


def load_1file(dump_file,  score_length = 100, filter_positive = True):
    # labels, array(), states, array(n, 70)
    with open(dump_file, 'rb') as f:
        dumps = six.moves.cPickle.load(f)
    labels=[]
    observations = []
    rewards = []
    actions = action_set_dict['default']
    actions_dict = {actions[i]: i for i in range(0, len(actions))}

    for dump in dumps:
        key = dump['debug']['action'][0]
        labels.append(actions_dict[key])
        rewards.append(dump["reward"])
        observations.append(dump['observation'])
        if 'frame' in dump['observation']:
            print(dump['observation']['frame'])
    states = observation_sim(observations)

    labels = labels[-score_length:]
    states = states[-score_length:]


    if filter_positive:
        labels, states, rewards = filter_positives(labels, states, rewards, score_length)
    # for i in range(len(labels)):
    #     print(labels[i], rewards[i])
    return(np.array(labels[:], dtype=np.float32), np.array(states[:], dtype=np.float32))

def load_data(input_path, tag,  score_length = 50, filter_positive = True):
    files = os.listdir(input_path)[:]
    labels = []
    states = []
    for ifilename in files:
        if ifilename.startswith(tag) and os.path.getsize(input_path+ifilename) > 0:
            #print(ifile)
            ilabels, istates = load_1file(input_path + ifilename, score_length, filter_positive)
            if len(ilabels) > 0:
                labels.append(ilabels)
                states.append(istates)

    labels = np.concatenate(labels).flatten()
    states = np.concatenate(states)
    print(labels.shape)
    print(states.shape)
    return labels, states

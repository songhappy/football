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

def load_1file(dump_file):
    # labels, array(), states, array(n, 70)
    with open(dump_file, 'rb') as f:
        dumps = six.moves.cPickle.load(f)
    labels = []
    rewards = []
    observations = []
    actions = action_set_dict['default']
    actions_dict = {actions[i]: i for i in range(0, len(actions))}

    print(len(dumps))
    for dump in dumps:
        #print(dump.keys())
        keys = dump["observation"].keys()
        key = dump['debug']['action'][0]
        labels.append(actions_dict[key])
        rewards.append(dump["reward"])
        #print(actions_dict[key])
        #print(dump["observation"]["ball"])
        observations.append(dump['observation'])
        # if 'frame' in dump['observation']:
        #     print(dump['observation']['frame'])
        # for key in dump.keys():
        #     if key != "observation":
        #         # print(key)
                # print(dump[key])
        keys2 = ["left_agent_controlled_player","right_agent_controlled_player","game_mode","score","ball_owned_team","ball_owned_player","steps_left"]
        for key in keys2:
            pass
            # print(key)
            # print(dump['observation'][key])
        #sys.exit()
    states = observation_sim(observations)
    labels, states = filter_positives(labels, states, rewards, score_length)
    return(np.array(labels, dtype=np.float32), np.array(states, dtype=np.float32))

def filter_positives(labels, states, rewards, score_length):
    indices = []
    labels_out = []
    states_out = []
    for r in rewards:
        if (r == 1): indices.append(rewards.index(r))

    for i in indices:
        labels_out.extend(labels[i-score_length-1:i+1])
        states_out.extend(states[i-score_length-1:i+1])
    return labels_out, states_out






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
    input_path = "/home/arda/intelWork/projects/googleFootball/11vs11/dumpMar28/episode_done_20200328-211331286483.dump"
    #input_path = "/tmp/dumps/score_20200217-224036658118.dump"

    labels, states = load_1file(input_path)
    print(labels)
    print(states)
    print(len(labels))
    print(len(states))


if __name__ == '__main__':
    main()
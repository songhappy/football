from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six.moves.cPickle
from gfootball.env.football_action_set import *
from gfootball.env import config
from gfootball.env import football_env
from predictor import MovementPredictor
from preprocess import *

from gfootball.env import wrappers
import sys

import numpy as np
from collections import deque
import time
import os


TRAIN = True
RENDER = False
DUMP = False
model_path = "/home/arda/intelWork/projects/googleFootball/dumpFeb18_model"


def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/dumpFeb18_score/"

    players = ['agent:left_players=1']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': DUMP,
        'players': players,
        'real_time': True,
        'level': '11_vs_11_easy_stochastic',
    })

    actions_size = len(action_set_dict["default"])
    feature_size = 195
    if (TRAIN):
        labels, states = load_data(input_path,"score_")
        train_split = 580000
        val_split=10000
        train_labels = labels[0:train_split]; val_labels = labels[-val_split:-1]
        train_states= states[0:train_split]; val_states = states[-val_split:-1]

        agent = MovementPredictor(actions_size, [feature_size])
        begin = time.time()
        for nepoch in range(1, 15):
            agent.train(train_states, train_labels, 64, epoch=1)
            loss, acc = agent.evaluate(val_states, val_labels)
            print("epoch={}".format(nepoch), "loss:{}".format(loss), "acc:{}".format(acc))
        agent.save(model_path)
        end = time.time()
        print("total time training:{}".format((end-begin)/60))
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

                print("episode:{}".format(nepisode), "score:{}".format(obs[0]["score"]))
                episode_reward = 0
        print("win:{}".format(win), "lose:{}".format(lose))

if __name__ == '__main__':
    main()
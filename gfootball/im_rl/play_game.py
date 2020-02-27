from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six.moves.cPickle
from gfootball.env.football_action_set import *
from gfootball.env import config
from gfootball.env import football_env
from gfootball.imitation_learning.predictor import MovementPredictor
from gfootball.imitation_learning.preprocess import *

from gfootball.env import wrappers
import sys

import numpy as np
from collections import deque
import time
import os


TRAINIM = True
TRAINRL = True
DURATION = 600
RENDER = False
DUMP = False
model_path = "/home/arda/intelWork/projects/googleFootball/dumpFeb4_model_im"

def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/dumpFeb4/"

    players = ['agent:left_players=1']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': DUMP,
        'players': players,
        'real_time': True,
        'level': 'academy_run_to_score_with_keeper',
    })

    actions_size = len(action_set_dict["default"])
    feature_size = 83

    if(TRAINRL):
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
        while nepisode < 100000:
            feature = observation_sim(obs)
            action = agent.act(feature)
            obs, reward, done, info = env.step(action)
            state = observation_sim(obs)
            agent.memorize(state, action, reward, done)
            episode_reward = episode_reward + reward
            if done:
                print("episode:{}".format(nepisode), "episode_reward:{}".format(episode_reward))
                env.reset()
                if reward > 0 or reward < 0:
                    #pass
                    agent.train_rl()
                nepisode = nepisode + 1
                if(episode_reward >0): win = win + 1
                elif episode_reward <0: lose = lose + 1

                episode_reward = 0
        print("win:{}".format(win), "lose:{}".format(lose))

if __name__ == '__main__':
    main()
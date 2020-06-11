from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gfootball.intel.im.preprocess import *

import numpy as np

TRAINPG = True
RENDER = False
DUMP = False
model_path = "/home/arda/guoqiong/football/log_rl/model/"

def learn():
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

    from gfootball.intel.rl.single_agents import PGAgent
    agent = PGAgent(actions_size, [feature_size], model_path)

    env = football_env.FootballEnv(cfg)
    env = wrappers.CheckpointRewardWrapper(env)
    if RENDER:
        env.render()
    env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())

    nepisode = 0
    episode_reward = 0
    win = 0
    lose = 0
    while nepisode < 500000:
        feature = observation_sim(obs)
        action = agent.get_action(feature)
        obs, reward, done, info = env.step(action)
        episode_reward = episode_reward + reward
        if done:
            print("episode:{}".format(nepisode), "episode_reward:{}".format(episode_reward))
            env.reset()
            nepisode = nepisode + 1
            if (episode_reward > 0):
                win = win + 1
            elif episode_reward < 0:
                lose = lose + 1
            episode_reward = 0

        state = observation_sim(obs)
        agent.memorize(state, action, reward, done)
        if done:
            agent.train_rl()
    agent.save(model_path)

    print("win:{}".format(win), "lose:{}".format(lose))

if __name__ == '__main__':
    learn()

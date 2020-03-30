from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gfootball.intel.im.preprocess import *

import numpy as np

TRAINPG = True
DURATION = 600
RENDER = False
DUMP = False
model_path = "/home/arda/intelWork/projects/googleFootball/run_keeper/dumpFeb4_model_rl"

def main():
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

    if(TRAINPG):
        from gfootball.intel.rl.policy_gradient import MovementPredictor
        env = football_env.FootballEnv(cfg)
        if RENDER:
            env.render()
        env.reset()
        agent = MovementPredictor(actions_size, [feature_size])
        obs, reward, done, info = env.step(env.action_space.sample())
        nepisode = 0
        episode_reward = 0
        win = 0
        lose = 0
        while nepisode < 100000:
            feature = observation_sim(obs)
            action = agent.act(feature)
            obs, reward, done, info = env.step(action)
            episode_reward = episode_reward + reward
            if done:
                print("episode:{}".format(nepisode), "episode_reward:{}".format(episode_reward))
                env.reset()
                if reward > 0:
                    reward = np.array([1], dtype=float)
                elif reward == 0:
                    reward = np.array([-0.5], dtype=float)
                else:
                    reward = np.array([-1], dtype=float)
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
    main()
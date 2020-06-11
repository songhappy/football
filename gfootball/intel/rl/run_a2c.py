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
from gfootball.intel.rl.single_agents import A2CAgent
import numpy as np

TRAINPG = True
RENDER = False
DUMP = False

model_path = "/home/arda/guoqiong/football/log_rl/model/"

def learn():    # In case of CartPole-v1, maximum length of episode is 500
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
    agent = A2CAgent(action_size, [state_size], model_path)
    env = football_env.FootballEnv(cfg)
    env = wrappers.CheckpointRewardWrapper(env)
    obs, reward, done, info = env.step(env.action_space.sample())

    nepisode = 0
    episode_reward = 0
    win = 0
    lose = 0
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
                sys.stdout.flush()
                print("episode:{}".format(nepisode), "score:{}".format(score))
                env.reset()
                # if reward > 0:
                #     reward = np.array([1], dtype=float)
                # elif reward == 0:
                #     reward = np.array([-0.5], dtype=float)
                # else:
                #     reward = np.array([-1], dtype=float)
                nepisode = nepisode + 1
                if (score >= 1):
                    win = win + 1
                elif score < 1:
                    lose = lose + 1
                score = 0


        # save the model
        if nepisode % 50 == 0:
            agent.actor.save_weights(model_path+"/actor.h5")
            agent.critic.save_weights(model_path+"/critic.h5")
if __name__ == '__main__':
    learn()
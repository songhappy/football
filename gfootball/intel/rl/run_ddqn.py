from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import gym

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam



from gfootball.intel.im.preprocess import *
from gfootball.intel.rl.single_agents import DoubleDQNAgent
import numpy as np

TRAINPG = True
RENDER = False
DUMP = False
EPISODES = 100000

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
    agent = DoubleDQNAgent(state_size, action_size)
    env = football_env.FootballEnv(cfg)
    env = wrappers.CheckpointRewardWrapper(env)
    obs, reward, done, info = env.step(env.action_space.sample())

    nepisode = 0
    episode_reward = 0
    win = 0
    lose = 0
    for e in range(EPISODES):
        done = False
        score = 0
        state = observation_sim(obs)
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = observation_sim(next_state)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100

            # save the sample <s, a, r, s'> to the replay memory
            if abs(reward) > 0:
                agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
                agent.train_model()
            episode_reward += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                sys.stdout.flush()
                print("episode:{}".format(nepisode), "CheckPointReward:{}".format(reward))

                # every episode, plot the play time
                nepisode = nepisode + 1

                env.reset()

        # save the model
        if e % 100 == 0:
            agent.update_target_model()
            agent.model.save_weights("./save_model/ddqn/"+str(e)+"ddqn.h5")
if __name__ == '__main__':
    learn()
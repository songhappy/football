from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six.moves.cPickle
from gfootball.env.wrappers import Simple115StateWrapper

from absl import app
from absl import flags
from absl import logging


from gfootball.env import config
from gfootball.env import football_env
import sys

# last_state = env.step([])
# while
# action = agent.act()
# last_state, new reward = env,step(action)
#
def load_data(dump_file, env):
    with open(dump_file, 'rb') as f:
        str = six.moves.cPickle.load(f)
    print(len(str))
    print(str[0]['observation'])
    print(type(str[0]))
    sys.exit()

    wrapper = Simple115StateWrapper(env)
    observations=[]
    for x in str:
        observations.append(wrapper.observation(x))

    print(observations)
    #dictionaries = json.load(str)
    #print(dictionaries)



def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/dumps/episode_done_20191220-101211706616.dump"
    players = ['ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=/home/arda/intelWork/projects/googleFootball/trained/academy_run_to_score_with_keeper_v2']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': True,
        'players': players,
        'real_time': True,
        'level': 'academy_pass_and_shoot_with_keeper'
    })

    env = football_env.FootballEnv(cfg)
    obs, reward, done, score_reward = env.step([])
    wrapper = Simple115StateWrapper(env)
    print(obs)
    print(reward)
    print(score_reward)
    print("----------------------------------------------")
    #feature = wrapper.observation(obs)
    #print(feature)
   # sys.exit()

    load_data(input_path, env)

    #TODO, rewrite wrapper.observation or call it in the right way to get simple115 floats from observation

if __name__ == '__main__':
    main()
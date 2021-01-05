from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gfootball.intel.im.models import MovementPredictorZoo
#from gfootball.intel.im.models import MovementPredictorKeras
from gfootball.intel.im.preprocess import *
from matplotlib import pyplot as plt
from zoo.orca import init_orca_context, stop_orca_context
import time

TRAIN = True
RENDER = False
DUMP = False
Score_length = 50
filter_positive = False
model_path = "/home/arda/intelWork/projects/googleFootball/dumpSep22_model_zoo"


def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/dumpSep22_score/"

    players = ['agent:left_players=1']
    cfg = config.Config({
        'action_set': 'default',
        'dump_full_episodes': DUMP,
        'players': players,
        'real_time': False,
        'level': '11_vs_11_easy_stochastic',
        'reward_experiment': 'scoring,checkpoints'
    })

    actions_size = len(action_set_dict["default"])
    feature_size = 195
    sc = init_orca_context(cluster_mode="local", cores=8,memory="20g")
    if (TRAIN):
        labels, states = load_data(input_path,"score", score_length, filter_positive)
        print(labels.shape)
        print(states.shape)
        sys.exit()
        train_split = 1360000
        val_split=20000
        train_labels = labels[0:train_split]; val_labels = labels[-val_split:-1]
        train_states= states[0:train_split]; val_states = states[-val_split:-1]

        agent = MovementPredictorZoo(actions_size, [feature_size])
        begin = time.time()
        nepochs = 1
        agent.train(train_states, train_labels, 1024, epoch=nepochs, model_path=model_path)
        loss, acc = agent.evaluate(val_states, val_labels)
        print("loss", loss)
        print("acc", acc)
        agent.save(model_path+"/model.bigdl")
        end = time.time()
        print("total time training:{}".format((end-begin)/60))
    else:
        env = football_env.FootballEnv(cfg)
        env = wrappers.CheckpointRewardWrapper(env)
        if RENDER:
            env.render()
        env.reset()
        agent = MovementPredictorZoo(actions_size, [feature_size])
        agent.load(model_path +"/model.bigdl")
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
            #print("reward:", reward)
            episode_reward = episode_reward + reward
            if done:
                env.reset()
                nepisode = nepisode + 1
                if(episode_reward >0): win = win + 1
                elif episode_reward <0: lose = lose + 1

                print("episode:{}".format(nepisode), "score:{}".format(obs[0]["score"]))
                episode_reward = 0
        print("win:{}".format(win), "lose:{}".format(lose))
    stop_orca_context()
if __name__ == '__main__':
    main()
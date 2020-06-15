from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gfootball.intel.im.models import MovementPredictor
from gfootball.intel.im.preprocess import *

TRAIN = True
DURATION = 600
RENDER = False
DUMP = True
model_path = "/home/arda/intelWork/projects/googleFootball/run_keeper/dumpFeb4_model_im_2"

def main():
    input_path = "/home/arda/intelWork/projects/googleFootball/run_keeper/dumpFeb4/"

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
    if ( not TRAIN):
        labels, states = load_data(input_path,"_1_")
        train_split = 450000
        val_split=40000
        train_labels = labels[0:train_split]; val_labels = labels[-val_split:-1]
        train_states= states[0:train_split]; val_states = states[-val_split:-1]

        agent = MovementPredictor(actions_size, [feature_size])
        for nepoch in range(1, 15):
            agent.train(train_states, train_labels, 64, epoch=1)
            loss, acc = agent.evaluate(val_states, val_labels)
            print("epoch={}".format(nepoch), "loss:{}".format(loss), "acc:{}".format(acc))
        agent.save(model_path)
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
            # print("--------------------")
            # print(action)
            # print(obs[0]["ball"])
            episode_reward = episode_reward + reward
            if done:
                env.reset()
                nepisode = nepisode + 1
                if(episode_reward >0): win = win + 1
                elif episode_reward <0: lose = lose + 1

                print("episode:{}".format(nepisode), "episode_reward:{}".format(episode_reward))
                episode_reward = 0
        print("win:{}".format(win), "lose:{}".format(lose))

if __name__ == '__main__':
    main()
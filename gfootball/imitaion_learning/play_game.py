import json
import six.moves.cPickle

# last_state = env.step([])
# while
# action = agent.act()
# last_state, new reward = env,step(action)
#
def load_data(dump_file):
    with open(dump_file, 'rb') as f:
        str = six.moves.cPickle.load(f)

    print(len(str))
    #dictionaries = json.load(str)
    #print(dictionaries)



def main():
    input_path = "/Users/guoqiong/intelWork/projects/googleFootball/dumps/episode_done_20191220-101211706616.dump"
    load_data(input_path)

if __name__ == '__main__':
    main()
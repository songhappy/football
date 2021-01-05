from gfootball.env import config
from gfootball.env import observation_preprocessing

import gfootball.env as football_env
from baselines.common.policies import build_policy

def dist2gate(position):
    distance = ((1-position[0])**2 + position[1]**2)**(0.5)
    return distance

def seg_reward(reward):
    if -0.1 < reward <= 0.005:
        n_reward = -0.1
    elif 0.005 < reward < 0.01:
        n_reward = 0
    elif 0.01 <= reward < 0.1:
        n_reward = 0.1
    else:
        n_reward = round(reward, 3)
    return n_reward


def shape_reward(reward, obs):

    return




def create_env(cfg_values):
    """Creates gfootball environment."""
    c = config.Config(cfg_values)
    env = football_env.football_env.FootballEnv(c)
    channel_dimensions = (
        observation_preprocessing.SMM_WIDTH,
        observation_preprocessing.SMM_HEIGHT)
    number_of_left_players_agent_controls=1
    number_of_right_players_agent_controls=0
    env =football_env._apply_output_wrappers(
        env, cfg_values['rewards'], cfg_values["representation"], channel_dimensions,
        (number_of_left_players_agent_controls +
         number_of_right_players_agent_controls == 1), cfg_values['stacked'])

    return env

# TODO, organize model_cfg to let more parameters pass
def create_model_ppo2(model_cfg, env, **network_kwargs):
    network = model_cfg['network']
    nsteps = model_cfg['nsteps']
    nminibatches=model_cfg['nminibatches']
    model_fn = model_cfg['model_fn']
    ent_coef= model_cfg['ent_coef']
    vf_coef= model_cfg['vf_coef']
    max_grad_norm=model_cfg['max_grad_norm']
    comm = model_cfg['comm']
    mpi_rank_weight=model_cfg['mpi_rank_weight']
    nenvs = model_cfg['nenvs']

    policy = build_policy(env, network, **network_kwargs)

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    return model

import psutil
import gc
def auto_garbage_collect(pct=0.7):
    if psutil.virtual_memory().percent >= pct:
        print("call gc ")
        gc.collect()
    return

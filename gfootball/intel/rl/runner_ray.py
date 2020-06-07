import numpy as np
from baselines.common.runners import AbstractEnvRunner
import ray
from gfootball.env import config
from gfootball.env import observation_preprocessing
from baselines.common.tf_util import get_session

import gfootball.env as football_env
import tensorflow as tf
from baselines.common.policies import build_policy

def create_env(cfg_values):
    """Creates gfootball environment."""
    c = config.Config(cfg_values)
    env = football_env.football_env.FootballEnv(c)
    channel_dimensions = (
        observation_preprocessing.SMM_WIDTH,
        observation_preprocessing.SMM_HEIGHT)
    number_of_left_players_agent_controls=1
    number_of_right_players_agent_controls=0
    rewards="scoring"
    representation='extracted'
    stacked=True
    env =football_env._apply_output_wrappers(
        env, rewards, representation, channel_dimensions,
        (number_of_left_players_agent_controls +
         number_of_right_players_agent_controls == 1), stacked)

    return env

def create_model(model_cfg, env, **network_kwargs):
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


from pympler.tracker import SummaryTracker
from pympler import muppy, summary

@ray.remote(memory=2500 * 1024 * 1024)
class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env_cfg, model_cfg, nsteps, gamma, lam):
        env = create_env(env_cfg)

        env.reset()
        model = create_model(model_cfg, env)

        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.s1 = summary.summarize(muppy.get_objects(remove_dups=False, include_frames=True))

    def update_model(self, param_vals):
        sess = get_session()
        params = tf.trainable_variables('ppo2_model')
        for var, val in zip(params, param_vals):
            update_placeholder = tf.placeholder(var.dtype, shape=var.get_shape())
            assign = var.assign(update_placeholder)
            sess.run(assign, {update_placeholder: val})

        del(params)
        del(param_vals)
        #self.print_num_of_total_parameters(params)

    def print_num_of_total_parameters(self, params):
        total_parameters = 0
        parameters_string = ""

        for variable in params:

            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            if len(shape) == 1:
                parameters_string += ("%s %d, " % (variable.name, variable_parameters))
            else:
                parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

        print(parameters_string)
        print("Total %d variables, %s params" % (len(params), "{:,}".format(total_parameters)))


    def run(self, params_id):
        # Here, we init the lists that will contain the mb of experiences
        self.update_model(params_id)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            if self.dones:
                self.env.reset()
           #print("infos", infos)

            for info in [infos]:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs,  dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape((self.nsteps, 1))
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape((self.nsteps, 1))
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        res = [*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos]

        #s2 = summary.summarize(muppy.get_objects(remove_dups=False, include_frames=True))
        #tracker.print_diff(self.s1, s2)


        return res
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])



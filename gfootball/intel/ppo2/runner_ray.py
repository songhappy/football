import numpy as np
from baselines.common.runners import AbstractEnvRunner
import ray
from baselines.common.tf_util import get_session

import tensorflow as tf
from gfootball.intel.utils import create_env, create_model_ppo2

from pympler import muppy, summary, asizeof
import sys
import numpy
#@ray.remote(memory=2500 * 1024 * 1024)

@ray.remote(memory=1000 * 1024 * 1024)
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
        model = create_model_ppo2(model_cfg, env)

        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.s1 = summary.summarize(muppy.get_objects(remove_dups=False, include_frames=True))
        self.assign = None
        self.update_placeholder = None
        # self.mb_ob =[]
        # self.mb_rewards = []
        # self.mb_actions = []
        # self.mb_values = []
        # self.mb_dones = []
        # self.mb_neglogpacs = []

    def update_model(self, param_vals):
        sess = get_session()
        params = tf.trainable_variables('ppo2_model')

        for var, val in zip(params, param_vals):
            self.update_placeholder = tf.placeholder(var.dtype, shape=var.get_shape())
            self.assign = var.assign(self.update_placeholder)
            sess.run(self.assign, {self.update_placeholder: val})

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
        numpy.set_printoptions(threshold=sys.maxsize)

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
 #           if rewards > 0: print(rewards)
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



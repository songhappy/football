# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs football_env on OpenAI's ppo2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from absl import app
from absl import flags
from baselines import logger
from baselines.bench import monitor
import gfootball.env as football_env

from gfootball.intel.ppo2 import ppo2_ray
import ray

FLAGS = flags.FLAGS

flags.DEFINE_string('address', '',
                    'Ip address of head node with port')


flags.DEFINE_string('level', 'academy_run_to_score_with_keeper',
                    'Defines type of problem being solved')
flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                 'extracted_stacked'],
                  'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring,checkpoints',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                    'gfootball_impala_cnn'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(2e6),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('ncpu', 8,'Number of cpus to run in parallel.')
flags.DEFINE_integer('nsteps', 128, 'Number of environment steps per epoch; '
                     'batch size is nsteps * nenv')
flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
flags.DEFINE_integer('nminibatches', 8,
                     'Number of minibatches to split one epoch to.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
flags.DEFINE_float('gamma', 0.993, 'Discount factor')
flags.DEFINE_float('cliprange', 0.27, 'Clip range')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm (clipping)')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')


env_cfg = {
    'level': '11_vs_11_easy_stochastic',
    'action_set': 'default',
    'dump_full_episodes': False,
    'dump_scores': False,
    'players': ['agent:left_players=1,right_players=0'],
    'dump_frequency': 50,
    'logdir': logger.get_dir(),
    'real_time': False,
    'render': False,
    'stacked': False,
    'rewards': "scoring,checkpoints",
    'representation': 'extracted'
}

def train(_):
  """Trains a PPO2 policy."""
  #vec_env = [create_single_football_env(i) for i in range(FLAGS.num_envs)]

  # Import tensorflow after we create environments. TF is not fork sake, and
  # we could be using TF as part of environment if one of the players is
  # controled by an already trained model.
  import tensorflow.compat.v1 as tf
  #ncpu = multiprocessing.cpu_count()
  ncpu = FLAGS.ncpu
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()

  import  time
  start = time.time()

  ppo2_ray.learn(address=FLAGS.address,
                 nenvs=FLAGS.num_envs,
                 network=FLAGS.policy,
                 total_timesteps=FLAGS.num_timesteps,
                 env_cfg=env_cfg,
                 seed=FLAGS.seed,
                 nsteps=FLAGS.nsteps,
                 nminibatches=FLAGS.nminibatches,
                 noptepochs=FLAGS.noptepochs,
                 max_grad_norm=FLAGS.max_grad_norm,
                 gamma=FLAGS.gamma,
                 ent_coef=FLAGS.ent_coef,
                 lr=FLAGS.lr,
                 logdir=logger.get_dir(),
                 log_interval=10,
                 save_interval=FLAGS.save_interval,
                 cliprange=FLAGS.cliprange,
                 load_path=FLAGS.load_path)

  end = time.time()
  print("************************", str(end - start))
if __name__ == '__main__':
  app.run(train)

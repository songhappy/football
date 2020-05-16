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


"""Script allowing to play the game by multiple players."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging


from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import wrappers
from baselines.common.runners import AbstractEnvRunner
import ray
FLAGS = flags.FLAGS

flags.DEFINE_string('players', 'keyboard:left_players=1',
                    'Semicolon separated list of players, single keyboard '
                    'player on the left by default')
flags.DEFINE_string('level', 'academy_pass_and_shoot_with_keeper', 'Level to play')
flags.DEFINE_enum('action_set', 'default', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
                  'If true, environment will slow down so humans can play.')
flags.DEFINE_bool('render', False, 'Whether to do game rendering.')

@ray.remote
class Runner:
    def __init__(self, cfg):

        # Lambda used in GAE (General Advantage Estimation)
        cfg = config.Config(cfg)
        self.env = football_env.FootballEnv(cfg)
        self.env.reset()

    def run(self):
        obs, reward, done, info = self.env.step([])
        return [obs, reward, done, info]

def main(_):
  players = FLAGS.players.split(';') if FLAGS.players else ''
  print(players)
  exit(1)

  assert not (any(['agent' in player for player in players])
             ), ('Player type \'agent\' can not be used with play_game.')
  cfg_values = {
      'action_set': 'default',
      'dump_full_episodes': True,
      'dump_scores': True,
      'players': players,
      'real_time': False,
      'render': False,
      'level': '11_vs_11_easy_stochastic',
  }

  ray.init()
  import time
  begin = time.time()
  duration = 10 * 60
  try:
      current = time.time()
      while current < begin + duration :
        runners = [Runner.remote(cfg_values) for _ in range(8)]
        out = [r.run.remote() for r in runners]
        out = ray.get(out)
        #print(out[0])

  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    exit(1)


if __name__ == '__main__':
  app.run(main)

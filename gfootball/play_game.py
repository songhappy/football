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

FLAGS = flags.FLAGS

flags.DEFINE_string('players', 'keyboard:left_players=1',
                    'Semicolon separated list of players, single keyboard '
                    'player on the left by default')
flags.DEFINE_string('level', 'academy_pass_and_shoot_with_keeper', 'Level to play')
flags.DEFINE_enum('action_set', 'default', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
                  'If true, environment will slow down so humans can play.')
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')


def main(_):
  players = FLAGS.players.split(';') if FLAGS.players else ''
  players = [
      'ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=/home/arda/intelWork/projects/googleFootball/trained/academy_run_to_score_with_keeper_v2']

  assert not (any(['agent' in player for player in players])
             ), ('Player type \'agent\' can not be used with play_game.')
  cfg = config.Config({
      'action_set': 'default',
      'dump_full_episodes': True,
      'players': players,
      'real_time': True,
      'level': 'academy_pass_and_shoot_with_keeper',
  })

  env = football_env.FootballEnv(cfg)
  if FLAGS.render:
    env.render()
  env.reset()
  total_reward = 0
  try:
    nepisode = 0
    while nepisode < 3000:
      obs, reward, done, info = env.step([])
      total_reward = total_reward + reward
      if done:
        env.reset()
        nepisode = nepisode + 1
    print("total_episodes: ", nepisode)
    print("total_reward: ", total_reward)
    #print("total_checkpoint_reward: " + total_checkpoint_reward)
  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    env.write_dump('shutdown')
    exit(1)


if __name__ == '__main__':
  app.run(main)

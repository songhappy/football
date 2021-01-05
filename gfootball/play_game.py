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

  assert not (any(['agent' in player for player in players])
             ), ('Player type \'agent\' can not be used with play_game.')
  cfg = config.Config({
      'action_set': 'default',
      'dump_full_episodes': True,
      'dump_scores': True,
      'players': players,
      'real_time': False,
      'render': True,
      'level': FLAGS.level,
      'display_game_stats':True,
      'video_quality_level': 1,  # 0 - low, 1 - medium, 2 - high
      'write_video': True
  })

  env = football_env.FootballEnv(cfg)
  if FLAGS.render:
    env.render()
  env.reset()
  episode_reward_p = 0
  episode_reward_n = 0
  import time
  begin = time.time()
  try:
    nepisode = 0
    step = 0
    while nepisode < 1:
      step = step + 1
      obs, reward, done, info = env.step([])
      if done:
        nepisode = nepisode + 1
        print("episode:{}".format(nepisode), "score:{}".format(obs["score"]))
        env.reset()
    end = time.time()
    print("total time: %.3f" % ((end-begin)/60), "min")
  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    env.write_dump('shutdown')
    exit(1)


if __name__ == '__main__':
  app.run(main)

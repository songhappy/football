from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from baselines import logger

from gfootball.intel.ddqn import ddqn_ray

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
flags.DEFINE_enum('policy', 'mlp', ['mlp'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(2e7),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 16, 'Number of environment steps per epoch; '
                     'batch size of runner is nsteps * nenv')
flags.DEFINE_integer('batch_size', 64, 'Number of environment steps for trainer')
flags.DEFINE_integer('memo_size', 2000, 'Number of environment steps for memory')
flags.DEFINE_integer('noptepochs', 16, 'Number of updates per epoch.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate')
flags.DEFINE_float('gama', 0.993, 'Discount factor')
flags.DEFINE_float('epsilon_min', 0.01, 'minimal epsilon')
flags.DEFINE_float('epsilon_decay', 0.9, 'epsilon_decay')

flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')



env_cfg = {
        'level': 'academy_run_to_score_with_keeper',
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
        'representation': 'raw'
    }

def train(_):
  import  time
  start = time.time()

  ddqn_ray.learn(address=FLAGS.address,
                 nenvs=FLAGS.num_envs,
                 network=FLAGS.policy,
                 total_timesteps=FLAGS.num_timesteps,
                 env_cfg= env_cfg,
                 state_size=83,
                 nsteps=FLAGS.nsteps,
                 batch_size=FLAGS.batch_size,
                 memo_size=FLAGS.memo_size,
                 noptepochs=FLAGS.noptepochs,
                 gama=FLAGS.gama,
                 lr=FLAGS.lr,
                 epsilon_min=FLAGS.epsilon_min,
                 epsilon_decay=FLAGS.epsilon_decay,
                 train_start=1000,
                 logdir=logger.get_dir(),
                 log_interval=10,
                 save_interval=FLAGS.save_interval
                 )

  end = time.time()
  print("************************", str(end - start))
if __name__ == '__main__':
  app.run(train)

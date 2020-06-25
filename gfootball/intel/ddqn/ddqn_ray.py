import os
import os.path as osp
import numpy as np
from baselines.common.runners import AbstractEnvRunner
import ray
from gfootball.env.football_action_set import *
from gfootball.intel.utils import create_env, create_model_ppo2
from baselines import logger
from pympler import muppy, summary, asizeof
from gfootball.intel.ddqn.agents_keras import DoubleDQNAgent
from collections import deque
import time
from gfootball.intel.im.preprocess import observation_sim

@ray.remote(memory=2000 * 1024 * 1024)
class Runner():
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env_cfg, model_cfg, nsteps, state_size, action_size):
        env = create_env(env_cfg)
        #self.s1 = summary.summarize(muppy.get_objects(remove_dups=False, include_frames=True))
        self.obs = env.reset()
        self.env = env
        self.nsteps =nsteps
        self.state_size = state_size
        self.gama = model_cfg['gama']
        self.agent = DoubleDQNAgent(state_size, action_size, network=model_cfg['network'],
                                    gama=model_cfg['gama'], lr=model_cfg['lr'], epsilon_min=model_cfg['epsilon_min'],
                                    epsilon_decay=model_cfg['epsilon_decay'], batch_size=model_cfg['batch_size'], train_start=model_cfg['train_start'])

    def run(self, params):
        self.agent.model.set_weights(params)

        memory = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            state = observation_sim(self.obs)
            # state = np.reshape(self.obs, [1, self.state_size])
            action = self.agent.get_action(state)
            self.obs, reward, done, _ = self.env.step(action)
            next_state = observation_sim(self.obs)
            #next_state = np.reshape(self.obs, [1, self.state_size])
            memory.append([state, action, reward, next_state, done])

            if done:
                self.obs = self.env.reset()

        # s2 = summary.summarize(muppy.get_objects(remove_dups=False, include_frames=True))
        # from pympler.tracker import SummaryTracker
        # tracker = SummaryTracker()
        # tracker.print_diff(self.s1, s2)
        # print(asizeof.asizeof(memory))
        # print(asizeof.asizeof(self.env))
        # print('agent', asizeof.asizeof(self.agent))
        del(params)
        return memory

def learn(*, address, nenvs, network="mlp", env_cfg, total_timesteps, state_size=115, nsteps=16,
          batch_size=64, memo_size=2000, noptepochs=4, gama=0.999, lr=0.0008, epsilon_min=0.01, epsilon_decay=0.9, train_start=1000,
          logdir=logger.get_dir(), log_interval=None, save_interval=None):

    action_size = len(action_set_dict["default"])
    model_cfg = {
        'nenvs':nenvs,
        'network': network,
        'gama': gama,
        'lr':lr,
        'epsilon_min': epsilon_min,
        'epsilon_decay':epsilon_decay,
        'batch_size': batch_size,
        'train_start':train_start
    }

    agent = DoubleDQNAgent(state_size, action_size, network=network,
                           gama=gama, lr=lr, epsilon_min=epsilon_min,
                           epsilon_decay=epsilon_decay, batch_size=batch_size, train_start=train_start)


    ray.init(address=address, lru_evict=True)
    params = agent.model.get_weights()
    params_id = ray.put(params)
    runners = []
    for i in range(nenvs):
        runner = Runner.remote(env_cfg=env_cfg, model_cfg=model_cfg, nsteps=nsteps, state_size=state_size, action_size=action_size)
        runners.append(runner)

    nbatch = nenvs * nsteps

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    memory = deque(maxlen=memo_size)

    for update in range(nupdates):
        tstart = time.perf_counter()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        results = [r.run.remote(params_id) for r in runners]
        results = ray.get(results)

        for result in results:
            for ele in result:
                memory.append(ele)

        runner_time = time.perf_counter()
        agent.train_model(memory=memory, k=noptepochs)
        agent.update_target_model()
        params = agent.model.get_weights()
        params_id = ray.put(params)

        tnow  = time.perf_counter()
        fps = int(nbatch / (tnow-tstart))
        if update % log_interval == 0 or update == 1:
            logger.logkv("misc/serial_timesteps", update * nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            logger.logkv('misc/update', tnow - tstart)
            logger.logkv('misc/runner', runner_time - tstart)
            logger.logkv('misc/trainer', tnow - runner_time)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logdir:
            checkdir = osp.join(logdir, 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            agent.save(savepath)
    return agent
This directory is built for running and training google football using analytics zoo.
# code structure
1. baselines are copied from baselines project for debugging purpose, no new develepment here.
2. im, using supervised learning
   run2score trains and plays a supervised model for senario of academy_run_to_score_with_keeper
   game11vs11 trains and plays a supervised model for senario of 11_vs_11_easy_stochastic
3. rl, reinforcement leraning approach
   3.1 simple_agents contains agents using keras on local mode, have tested A2C, PolicyGradient, and DDQN.
        hyperparameters of ddqn
        rl = 0.00008
        batch_size = 
   3.2 distribute ppo2 using ray and analytics zoo
       run_ppo2_ray distribut runner_ray and ppo2_ray
   
4. im_rl, using supervised learning and reinforcement learning


 

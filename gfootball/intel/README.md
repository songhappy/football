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
       steps to run distrbuted actors for ppo2 using ray
       step1, install google football related tools on each node following guide on [google football github](https://github.com/google-research/football)
       step2, install ray and setup ray cluster connect nodes using ray following guide on [ray setup](https://docs.ray.io/en/latest/using-ray-on-a-cluster.html#manual-cluster-setup)
       step3, run the script
            ```bash
            python3 run_ppo2_ray.py
            ```
   
4. im_rl, using supervised learning and reinforcement learning

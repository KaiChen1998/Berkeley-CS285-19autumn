# Problem 1: Dynamic Model

```bash
python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1

python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1

python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1
```

- Case1: train_steps_per_iteration: 500, n_layers: 1, size: 32

  ![img](cs285/data/mb_cheetah_n500_arch1x32_cheetah-cs285-v0_09-03-2020_16-47-16/itr_0_losses.png)

![img](cs285/data/mb_cheetah_n500_arch1x32_cheetah-cs285-v0_09-03-2020_16-47-16/itr_0_predictions.png)

- Case2: train_steps_per_iteration: 5, n_layers: 2, size: 250

  ![img](cs285/data/mb_cheetah_n5_arch2x250_cheetah-cs285-v0_09-03-2020_16-51-53/itr_0_losses.png)

  ![img](cs285/data/mb_cheetah_n5_arch2x250_cheetah-cs285-v0_09-03-2020_16-51-53/itr_0_predictions.png)

- Case2: train_steps_per_iteration: 500, n_layers: 2, size: 250

  ![img](cs285/data/mb_cheetah_n500_arch2x250_cheetah-cs285-v0_09-03-2020_16-52-11/itr_0_losses.png)

  ![img](cs285/data/mb_cheetah_n500_arch2x250_cheetah-cs285-v0_09-03-2020_16-52-11/itr_0_predictions.png)

- Conclusion: when only training with a fixed dataset, the model will stack easily.



# Problem 2: Action Selection

```bash
python cs285/scripts/run_hw4_mb.py --exp_name obstacles_singleiteration --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10  --video_log_freq -1
```

- Eval_AverageReturn: -36.20
- Train_AverageReturn: -167.1

![img](cs285/data/mb_obstacles_singleiteration_obstacles-cs285-v0_09-03-2020_17-09-54/itr_0_losses.png)

![img](cs285/data/mb_obstacles_singleiteration_obstacles-cs285-v0_09-03-2020_17-09-54/itr_0_predictions.png)



#  Problem 3: MBRL

```bash
python cs285/scripts/run_hw4_mb.py --exp_name obstacles --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name reacher --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name cheetah --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20 --video_log_freq -1 -gpu
```

![img](fig/obstacles.jpg)

![img](fig/reacher.jpg)

![img](fig/cheetah.png)



# Problem 4: Hyperparameter

- MPC planning horizon

```bash
python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon5 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon15 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon30 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 -gpu
```

![img](fig/horizon.jpg)



- random shooting sample numer

```bash
python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_numseq100 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_numseq1000 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 1000 --video_log_freq -1 -gpu
```

![img](fig/random_sample.jpg)



- number of ensemble model

```bash
python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble1 --env_name reacher-cs285-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble3 --env_name reacher-cs285-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 -gpu

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble5 --env_name reacher-cs285-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 -gpu
```

![img](fig/ensemble.jpg)

- Conclusion:
  - using MPC can decrease need for longer horizon planning
  - when using random shooting method to choose the best action sequence, more samples can benefit
  - More ensemble more can benefit
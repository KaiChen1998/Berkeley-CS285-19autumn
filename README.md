# CS285 Deep Reinforcement Learning @ Berkeley

## Homework 1

- Behavior Cloning: Normal Supervised Learning
- DAgger: improve the performance of SL a lot
- Transition-based: no need to keep trajectories



## Homework 2

- Policy Gradient with reward-to-go Q function and value function based baseline
- Value function baseline using MC estimation
- Also we use normalization baseline here
- Discount factor
- GAE: Generalized Advantage Estimation



## Homework 3

- Deep Q learning:
  - Vanilla DQN
  - Double DQN
  - DQN with polyak average
  - ε-greedy exploration policy
- Actor-Critic
  - bootstrap estimation
  - when doing bootstrap, how often do we update target (no too often considering moving target, and no too infrequent considering bootstrap is still a biased estimation)



## Homework 4:

- Model-based RL with model predictive control (MPC)
  - transition model with residual connection
  - open loop decision based on random shooting and expected reward bootstrap ensemble 
  - MPC: but only take the first action from the best action sequence
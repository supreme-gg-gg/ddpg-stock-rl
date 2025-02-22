# Deep Reinforcement Learning for Portfolio Management

A modern PyTorch implementation of the paper [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059).

The original code in Tensorflow 1.x can be found [here](https://github.com/vermouth1992/drl-portfolio-management).

## Algorithms used in this work

- Deep Deterministic Policy Gradient using EIIE (Ensemble of Identical Independent Evaluators) architecture (CNN and LSTM)

- Imitation Learning (work in progress) using CNN and LSTM as benchmark

## Dataset

- We use dataset from kaggle. It can be found [here](https://www.kaggle.com/camnugent/sandp500)

## Reference

- The environment is inspired by https://github.com/wassname/rl-portfolio-management
- DDPG implementation is inspired by http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

#!/usr/bin/env python3

import gym
import pybullet, pybullet_envs
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('AntBulletEnv-v0')
env.render(mode = 'human')

MAX_AVERAGE_SCORE = 271
LEARNING_RATE = 3e-3
LOOP_NUM = int(8e3) # original: 8e3
TIMESTEPS = int(1e4) # original: 1e4
SAVE_PATH = 'ppo_ant_saved_model'
N_EVAL_EPISODES = 5

policy_kwargs = {
    'activation_fn' : th.nn.LeakyReLU,
    'net_arch' : [512, 512],
}

def main() -> None:
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate = LEARNING_RATE,
        policy_kwargs = policy_kwargs,
        verbose = 1
    )

    for ind in range(LOOP_NUM):
        print(f'Loop: {ind}')
        model.learn(total_timesteps = TIMESTEPS)
        model.save(SAVE_PATH)
        mean_reward, std_reward = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes = N_EVAL_EPISODES,
        )
        print(f'MeanRw: {mean_reward}')
        if mean_reward >= MAX_AVERAGE_SCORE:
            break

if __name__ == '__main__':
    main()


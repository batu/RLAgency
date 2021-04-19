import os

import gym
import numpy as np
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rlnav import UnityToGymMultipleWrapper

ENV_PATH = r"C:\Users\user\Desktop\RLNav\NavigationEnvironments\P0\ProfileServer20\Env.exe"
unity_env = UnityEnvironment(file_name=None, base_port=6008)

env = UnityToGymMultipleWrapper(unity_env)
# env = Monitor(env, "tmp/")
# observation = env.reset()

model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=50000)

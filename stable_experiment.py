import gym
import numpy as np
import torch

from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

ENV_PATH = r"C:\Users\user\Desktop\RLNav\RLAgency\Builds\P0\BasicSingle\Env.exe"

unity_env = UnityEnvironment(ENV_PATH, base_port=6008)
# unity_env.reset()


env = UnityToGymWrapper(unity_env)
observation = env.reset()

# model = PPO("MlpPolicy", env, verbose=2)
model = PPO.load(r"C:\Users\user\Desktop\RLNav\RLAgency\Results\0_Pipeline\Mutliple\PPO\SB4.zip")
model.set_env(env)
model.learn(total_timesteps=50000)
model.save(r"C:\Users\user\Desktop\RLNav\RLAgency\Results\0_Pipeline\Mutliple\PPO\SB4")

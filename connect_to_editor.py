from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from mlagents_envs.exception import UnityTimeOutException
import torch as th

from mlagents_envs.base_env import (
    ActionTuple,
    DecisionSteps,
    DecisionStep,
    TerminalSteps,
    TerminalStep,
)

import os, yaml
import random
import time
import datetime

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper, UnityToMultiGymWrapper 
import wandb
from rlnav.logging import WANDBMonitor, test_model
from rlnav.schedules import linear_schedule
from pathlib import Path
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel



DEBUG = True
PROTOTYPE_NAME = "Jump"
ENV_NAME = "Baseline_8"
EXPERIMENT_NAME = f"Representative"

alg   = "SAC" # "PPO"
TREATMENT_NAME = f"{alg}"

with open(Path(f"rlnav/configs/{alg}_rlnav_defaults.yaml"), 'r') as f:
  alg_config = yaml.load(f, Loader=yaml.FullLoader)

if DEBUG: PROTOTYPE_NAME="DEBUG"



channel = EngineConfigurationChannel()
env_channel = EnvironmentParametersChannel()
env = UnityEnvironment(None, side_channels=[channel, env_channel])
channel.set_configuration_parameters(time_scale = 2.0)
env_channel.set_float_parameter("parameter_1", 2.0)

env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
for _ in range(1000):
  act = ActionTuple(np.array([np.random.uniform(-1,1,(3,))]), None)
  env.set_actions(behavior_name, act)
  ds = env.step()
  time.sleep(0.02)


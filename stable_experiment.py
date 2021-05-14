from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from mlagents_envs.exception import UnityTimeOutException
import torch as th

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




PROTOTYPE_NAME = "Jump"
alg   = "PPO" # "PPO"


PROTOTYPE_NAME = f"{PROTOTYPE_NAME}" 
ENV_NAME = "Baseline_8"
EXPERIMENT_NAME = f"Representative"
TREATMENT_NAME = f"{alg}"

with open(Path(f"rlnav/configs/{alg}_rlnav_defaults.yaml"), 'r') as f:
  alg_config = yaml.load(f, Loader=yaml.FullLoader)

base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_NAME}")
ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  

wandb_config = {
    "ENV_Name":ENV_NAME,
    "Treatment":TREATMENT_NAME,
    "Algorithm": alg,
    "Source":"Laptop",
}

if alg == "PPO":
  neural_network_config = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[512, 512], vf=[512, 512])])
  wandb_config["Activation"] = neural_network_config["activation_fn"]
  wandb_config["VF"] = neural_network_config["net_arch"][0]["vf"],
  wandb_config["PI"] = neural_network_config["net_arch"][0]["pi"],
elif alg == "SAC":
  neural_network_config = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[512, 512], qf=[512, 512]))
  wandb_config["Activation"] = neural_network_config["activation_fn"]
  wandb_config["VF"] = neural_network_config["net_arch"]["qf"]
  wandb_config["PI"] = neural_network_config["net_arch"]["pi"]

wandb_config.update(alg_config)

def make_env(rank, seed=0):
  def _init():
    unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + rank)
    env = UnityToMultiGymWrapper(unity_env)
    env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
    return env
  return _init

port_randomizer = random.randint(0,4000)
env = MultiAgentVecEnv(make_env(port_randomizer))

if alg == "SAC":
    model = SAC("MlpPolicy", env, policy_kwargs=neural_network_config, **alg_config)  
elif alg == "PPO":
    model = PPO("MlpPolicy", env, policy_kwargs=neural_network_config, **alg_config)

model.learn(total_timesteps=10000)
final_success_rate = test_model(env, model)
wandb.log({"Final Success Rate":final_success_rate})
try:
  dirpath = Path(f"Results/{PROTOTYPE_NAME}/{EXPERIMENT_NAME}/{TREATMENT_NAME}")
  os.makedirs(dirpath, exist_ok=True)
  model.save(dirpath / f"{PROTOTYPE_NAME}_{final_success_rate:.1%}.zip")
except Exception as e:
  print("Couldn't save.")
  print(e)
wandb.finish()

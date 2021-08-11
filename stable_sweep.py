from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy

from mlagents_envs.exception import UnityTimeOutException
import torch as th

import os, yaml
import random
import time
import datetime

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper, UnityToMultiGymWrapper 
import wandb
from rlnav.logging import WANDBMonitor
from rlnav.schedules import linear_schedule
from pathlib import Path

PROTOTYPE_NAME = "Jump"
EXPERIMENT_NAME = f"RewardSweep"
base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_NAME}")


learning_rates = [("lin-4",linear_schedule(3e-4)), ("con-4", 3e-4)] 
environments = ["SSparse_16Agent", "SBaseline_8Agent", "SNoPBRS_8Agent", "SSparse_8Agent", "SBonzai_8Agent", "SFixPBRS_8Agents", "Baseline_8Agent"]

random.shuffle(environments)
random.shuffle(learning_rates)

for _ in range(100):
  try:
    for lrn, lr in learning_rates:
      for ENV_NAME in environments: 
        ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
        with open(Path(f"rlnav/configs/PPO_rlnav_defaults.yaml"), 'r') as f:
          alg_config = yaml.load(f, Loader=yaml.FullLoader)

        TREATMENT_NAME = f"{ENV_NAME}_LR{lrn}"

        wandb_config = {
            "ENV_Name":ENV_NAME,
            "Treatment":TREATMENT_NAME,
            "Algorithm": "PPO",
            "Source":"Laptop",
            "learning_rate": lrn
        }

        alg_config["learning_rate"] = lr
        neural_network_config = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[512, 512], vf=[512, 512])])
        wandb_config["Activation"] = neural_network_config["activation_fn"]
        wandb_config["VF"] = neural_network_config["net_arch"][0]["vf"]
        wandb_config["PI"] = neural_network_config["net_arch"][0]["pi"]

        wandb_config.update(alg_config)

        def make_env(rank, seed=0):
          def _init():
            unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + rank)
            env = UnityToMultiGymWrapper(unity_env)
            env = WANDBMonitor(env, wandb_config, prototype="{PROTOTYPE_NAME}", experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
            return env
          return _init

        port_randomizer = random.randint(0,4000)
        env = MultiAgentVecEnv(make_env(port_randomizer))
        model = PPO("MlpPolicy", env, policy_kwargs=neural_network_config, **alg_config)

        total_timesteps = 2_000_000 if ENV_NAME == "SSparse_16Agent" else 1_000_000
        model.learn(total_timesteps=total_timesteps)
        wandb.finish()
        env.close()
  except UnityTimeOutException as e:
    print("Unity timed out.")
    print(e)
    continue
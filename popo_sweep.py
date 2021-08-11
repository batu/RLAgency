from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy

from mlagents_envs.exception import UnityTimeOutException, UnityWorkerInUseException
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

PROTOTYPE_NAME = "SoloBuilding"
EXPERIMENT_NAME = f"Baseline"
base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_NAME}")


environments = ["Baseline505"]

random.shuffle(environments)
ts = 5
for _ in range(100):
  try:
    for ENV_NAME in environments: 
      ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
      with open(Path(f"rlnav/configs/PPO_rlnav_defaults.yaml"), 'r') as f:
        alg_config = yaml.load(f, Loader=yaml.FullLoader)

      TREATMENT_NAME = f"{ENV_NAME}_PPO"
     
      wandb.finish()
      wandb_config = {
          "ENV_Name":ENV_NAME,
          "Treatment":TREATMENT_NAME,
          "Algorithm": "PPO",
          "Source":"Laptop",
          "TimeScale":ts
      }

      neural_network_config = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[512, 512], vf=[1024, 512, 256])])
      wandb_config["Activation"] = neural_network_config["activation_fn"]
      wandb_config["VF"] = neural_network_config["net_arch"][0]["vf"]
      wandb_config["PI"] = neural_network_config["net_arch"][0]["pi"]

      alg_config["batch_size"] = 1024
      wandb_config.update(alg_config)

      def make_env(rank, seed=0):
        def _init():
          channel = EngineConfigurationChannel()
          unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + rank, side_channels=[channel])
          env = UnityToMultiGymWrapper(unity_env)
          env = WANDBMonitor(env, wandb_config, prototype=f"{PROTOTYPE_NAME}", experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
          channel.set_configuration_parameters(time_scale = ts)
          return env
        return _init

      port_randomizer = random.randint(0,4000)
      env = MultiAgentVecEnv(make_env(port_randomizer))
      model = PPO("MlpPolicy", env, policy_kwargs=neural_network_config, **alg_config)

      total_timesteps = 3_000_000
      model.learn(total_timesteps=total_timesteps)
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
      env.close()
  except UnityTimeOutException as e:
    print("Unity timed out.")
    print(e)
    continue
  except UnityWorkerInUseException as e:
    print("Unity timed out.")
    print(e)
    continue
  
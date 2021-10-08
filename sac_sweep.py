import random
import wandb
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import MultiAgentVecEnv
from stable_baselines3.common.buffers import DictReplayBuffer

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
from mlagents_envs.exception import UnityTimeOutException, UnityWorkerInUseException

from rlnav.custom_networks import SACCustomPolicy
from rlnav.logging import WANDBMonitor, test_model
from rlnav.utils import count_parameters
from rlnav.configs.configurations import setup_configurations
from rlnav.wrappers import ConvDictWrapper

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import yaml
import torch as th
import asyncio

PROTOTYPE_NAME = "Chubb"
EXPERIMENT_NAME = f"SAC"
PROTOTYPE_PATH_NAME = "Chubb"
base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_PATH_NAME}")


environments = ["Baseline"]

for _ in range(19):
  try:
    for envname in environments:
      ENV_NAME = envname
      ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
      TREATMENT_NAME = f"{ENV_NAME}"
      
      with open(Path("rlnav/configs/SAC_rlnav_config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

      config["environment_config"]["curriculum_length"] = 7_500_000
      config["observation_config"]["randomization_percentage"] = 0.15
      

      wandb_config, network_config, alg_config, channels = setup_configurations(config)
      wandb_config["ENV_Name"]  = PROTOTYPE_NAME
      wandb_config["Treatment"] = TREATMENT_NAME

      alg_config["gradient_steps"] = 128
      alg_config["learning_rate"] = 5e-4
      alg_config["learning_starts"] = 100_000
      alg_config["buffer_size"] = 1000_000

      network_config["net_arch"]["pi"] = [512]
      network_config["net_arch"]["qf"] = [1024]

      alg_config["replay_buffer_class"] = DictReplayBuffer

      def make_env():
        def _init():
          unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + random.randint(0,5000), side_channels=channels)
          env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
          env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
          env = ConvDictWrapper(env)
          return env
        return _init

      env = MultiAgentVecEnv(make_env())
      # model = SAC(SACCustomPolicy, env, policy_kwargs=network_config, **alg_config)
      model = SAC("MlpPolicy", env, policy_kwargs=network_config, **alg_config)
      count_parameters(model.policy)
      WANDBMonitor.model = model

      total_timesteps = 20_000_000
      model.learn(total_timesteps=total_timesteps)
    

      final_success_rate = test_model(env, model)
      wandb.log({"Final Success Rate":final_success_rate})

      try:
        model.save(WANDBMonitor.dirpath / f"FinalNetwork.zip")
      except Exception as e:
        print("Couldn't save.")
        print(e)

      env.close()
      wandb.finish()
  except UnityTimeOutException as e:
    print("Unity timed out.")
    print(e)
    continue
  except UnityWorkerInUseException as e:
    print("Unity timed out.")
    print(e)
    continue

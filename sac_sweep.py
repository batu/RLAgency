import random
import wandb
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import MultiAgentVecEnv

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
from mlagents_envs.exception import UnityTimeOutException, UnityWorkerInUseException

from rlnav.custom_networks import SACCustomPolicy
from rlnav.logging import WANDBMonitor, test_model
from rlnav.utils import count_parameters
from rlnav.configs.configurations import setup_configurations

import torch as th

PROTOTYPE_NAME = "Urban"
EXPERIMENT_NAME = f"Debug"
PROTOTYPE_PATH_NAME = "Urban"
base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_PATH_NAME}")

hyperparameter_list_1   = [True, False]
hyperparameter_list_2   = [8]
environments = [(32, "Debug")] 

for _ in range(5):
  try:
    for ratio_adjust in hyperparameter_list_2:
      for SDE in hyperparameter_list_1:
        for envcount, envname in environments:
          ENV_NAME = envname
          ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
          TREATMENT_NAME = f"{ENV_NAME}"
          
          wandb_config, network_config, alg_config, channels = setup_configurations("rlnav/configs/SAC_rlnav_config.yaml")
          wandb_config["ENV_Name"]  = PROTOTYPE_NAME
          wandb_config["Treatment"] = TREATMENT_NAME


          def make_env():
            def _init():
              # unity_env = UnityEnvironment(None)
              unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + random.randint(0,5000), side_channels=channels)
              env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
              env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
              return env
            return _init

          env = MultiAgentVecEnv(make_env())
          # model = SAC(SACCustomPolicy, env, policy_kwargs=network_config, **alg_config)
          model = SAC("MlpPolicy", env, policy_kwargs=network_config, **alg_config)
          count_parameters(model.policy)
          
          total_timesteps = 1_000_000
          model.learn(total_timesteps=total_timesteps)
          
          final_success_rate = test_model(env, model)
          wandb.log({"Final Success Rate":final_success_rate})


          try:
            model.save(WANDBMonitor.dirpath / f"Fin_{PROTOTYPE_NAME}_{final_success_rate:.1%}.zip")
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

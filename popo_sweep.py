from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy
from rlnav.configs.configurations import setup_configurations

from rlnav.custom_graphnetworks import GraphActorCriticPolicy
from mlagents_envs.exception import UnityTimeOutException, UnityWorkerInUseException
from rlnav.wrappers import GraphDictWrapper

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


PROTOTYPE_PATH_NAME = "Urban"       # Sometimes for legacy reasons the path and name are seperate.
PROTOTYPE_NAME      = "Graph"       # Whatever environment I am using
EXPERIMENT_NAME     = f"GNN_SweepLocal"
base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_PATH_NAME}")

environments = ["EasyBaseline"]

embedding_sizes = [16, 64, 128]
hidden_layers = [False, 3, 2]
hidden_widths = [128, 512, 1024]

random.shuffle(embedding_sizes)
random.shuffle(hidden_layers)
random.shuffle(hidden_widths)
network_structures = [((dict(vf=[512, 512], pi=[512, 512]),), "Seperate512")]


for _ in range(100):
  try:
    for env_count in embedding_sizes:
      for embedding_size in embedding_sizes:
        for width in hidden_widths:
          for hidden_layer in hidden_layers:
            for netarch, netname in network_structures:
              for ENV_NAME in environments: 
                ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
                with open(Path(f"rlnav/configs/PPO_rlnav_defaults.yaml"), 'r') as f:
                  config = yaml.load(f, Loader=yaml.FullLoader)

                config["environment_config"]["env_count"] = 32
                config["sac_config"]["batch_size"] = 256
                config["sac_config"]["n_epochs"] = 3
                config["sac_config"]["learning_rate"] = linear_schedule(3e-4)

                
                wandb_config, network_config, alg_config, channels = setup_configurations(config)

                netarch[0]["graph_hidden_layers_count"] = hidden_layer
                netarch[0]["graph_hidden_layers_width"] = width
                netarch[0]["graph_embedding_size"] = embedding_size
                
                network_config["net_arch"] = netarch

                if hidden_layer:
                  TREATMENT_NAME = f"GraphNN_GLayeres{hidden_layer}_GWidth{width}_Emb{embedding_size}"
                  wandb_config["Graph"] = True
                else:
                  TREATMENT_NAME = f"Baseline"
                  wandb_config["Graph"] = False


                if hidden_layer:
                  def make_env():
                    def _init():
                      unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + random.randint(0,5000), side_channels=channels)
                      env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
                      env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
                      env = GraphDictWrapper(env)

                      return env
                    return _init

                  env = MultiAgentVecEnv(make_env())
                  model = PPO(GraphActorCriticPolicy, env, policy_kwargs=network_config, **alg_config)
                else:
                  def make_env():
                    def _init():
                      unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + random.randint(0,5000), side_channels=channels)
                      env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
                      env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
                      return env
                    return _init

                  env = MultiAgentVecEnv(make_env())
                  model = PPO("MlpPolicy", env, policy_kwargs=network_config, **alg_config)
                
                print(model.policy)
                total_timesteps = 10_000_000
                model.learn(total_timesteps=total_timesteps)
                final_success_rate = test_model(env, model)
                wandb.log({"Final Success Rate":final_success_rate})

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
  
import os


from mlagents_envs.logging_util import DEBUG
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import MultiAgentVecEnv

from rlnav.configs.configurations import setup_configurations, get_config_dict

from rlnav.custom_graphnetworks import AdamGCN, AggregateGCN, GraphActorCriticPolicy, SingleLayerGCN
from mlagents_envs.exception import UnityTimeOutException, UnityWorkerInUseException
from rlnav.wrappers import GraphDictWrapper

import random

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
import wandb
from rlnav.logging import WANDBMonitor, test_model
from rlnav.schedules import linear_schedule
from pathlib import Path

is_local = os.name == "nt"

PROTOTYPE_PATH_NAME = "Debug"       # Sometimes for legacy reasons the path and name are seperate.
PROTOTYPE_NAME      = "Debug"       # Whatever environment I am using
EXPERIMENT_NAME     = f"Sweep33"

if is_local:
  base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_PATH_NAME}")
else:
  base_bath = Path(fr"/workspace/NavigationEnvironments/Docker/{PROTOTYPE_PATH_NAME}")


environments = ["Baseline"]
nn_styles    = [AggregateGCN, SingleLayerGCN]#, AdamGCN]
local_stepss = [0,1,2,3]
edge_styles  = ["child_to_parent"]
agg_styles   = ["residual", "normal"]

random.shuffle(agg_styles)
random.shuffle(nn_styles)
random.shuffle(edge_styles)
random.shuffle(local_stepss)

network_structures = [((dict(vf=[512, 512], pi=[512, 512]),), "Seperate512")]
ENV_NAME = "Baseline"

for _ in range(100):
  try:
    for local_steps in local_stepss: 
      for agg_style in agg_styles:
        for edge_style in edge_styles:
          for nn_style in nn_styles:
            for netarch, netname in network_structures:
              if is_local:
                ENV_PATH = base_bath / fr"{ENV_NAME}/Env.exe"  
              else:
                ENV_PATH = base_bath / fr"{ENV_NAME}/Env.x86_64" 
                seed = random.randint(0,10000)
                print("Before chmod.")
                os.system(f"chmod -R 755 {ENV_PATH}")  

              config = get_config_dict(PPO)
              config["observation_config"]["use_occupancy"] = False
              config["observation_config"]["use_whiskers"] = False
              config["observation_config"]["use_depthmap"] = False

              config["environment_config"]["env_count"] = 32
              config["sac_config"]["ent_coef"] = 0.001

              config["sac_config"]["batch_size"] = 256
              config["sac_config"]["n_epochs"] = 3
              config["sac_config"]["learning_rate"] = linear_schedule(3e-4)

              wandb_config, network_config, alg_config, channels = setup_configurations(config)

              netarch[0]["global_steps"] = 4
              netarch[0]["local_steps"]   = local_steps
              netarch[0]["network_type"]  = nn_style
              netarch[0]["edge_style"]    = edge_style
              netarch[0]["agg_style"]     = agg_style

              netarch[0]["graph_hidden_layers_width"] = 16
              netarch[0]["graph_embedding_size"] = 8
              netarch[0]["curriculum_length"] = 0
              
              network_config["net_arch"] = netarch

              TREATMENT_NAME = f"{nn_style.__name__}_{agg_style}_Application{local_steps}"

              wandb_config["Graph"] = True
              wandb_config["Local"] = True
              wandb_config["Aggregation Method"] = agg_style
              wandb_config["Edge Style"] = edge_style
              wandb_config["Net Style"] = nn_style.__name__
              
              def make_env():
                def _init():
                  unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + random.randint(0,5000), side_channels=channels)
                  env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
                  env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
                  env.next_test_timestep=10_000_000
                  env = GraphDictWrapper(env)
                  return env
                return _init

              env = MultiAgentVecEnv(make_env())
              # model = PPO("MlpPolicy", env, policy_kwargs=network_config, **alg_config)
              model = PPO(GraphActorCriticPolicy, env, policy_kwargs=network_config, **alg_config)
              total_timesteps = 250_000
              model.learn(total_timesteps=total_timesteps)
              final_success_rate = test_model(env, model, test_count=640)
              wandb.log({"Final Success Rate":final_success_rate})

              wandb.finish()
              env.close()
              exit()
  except UnityTimeOutException as e:
    print("Unity timed out.")
    print(e)
    continue
  except UnityWorkerInUseException as e:
    print("UnityWorkerInUseException.")
    print(e)
    continue
  
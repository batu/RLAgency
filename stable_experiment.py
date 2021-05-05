from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy

import os
import random
import time
import datetime

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper, UnityToMultiGymWrapper 

from rlnav.logging import WANDBMonitor
from pathlib import Path

algs = ["SAC", "PPO"]
ens = ["Empty_8Agent_DF5", "Empty_1Agent_DF5"]
treats = ["Dummy", "Multi"]

random.shuffle(algs)
random.shuffle(ens)
random.shuffle(treats)



for _ in range(100):
    for alg in algs:
        for treat in treats:
            for en in ens:
                if treat == "Dummy" and en == "Empty_8Agent_DF5":
                    continue

                breaker = random.randint(0,4000)

                PROTOTYPE_NAME = "Benchmark"
                PROJECT_NAME = f"{datetime.datetime.today().day}_{PROTOTYPE_NAME}" #Jump
                ENV_NAME = "Ball3D_1Agent_1Frame"
                TREATMENT_NAME = treat

                base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROJECT_NAME}")
                ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
                GROUP_NAME = f"{alg}_{ENV_NAME}_{TREATMENT_NAME}"

                config = {
                    "ENV_PATH":ENV_PATH
                }

                def make_env(rank, seed=0):
                    def _init():
                        unity_env = UnityEnvironment(str(config["ENV_PATH"]), base_port=5172 + rank)
                        env = UnityToMultiGymWrapper(unity_env) if TREATMENT_NAME.lower() == "multi" else UnityToGymWrapper(unity_env) 
                        env = MonitorMulti(env, "tmp") if TREATMENT_NAME.lower() == "multi" else Monitor(env, "tmp") 
                        env = WANDBMonitor(env, config, project=PROJECT_NAME, group=GROUP_NAME)
                        return env
                    return _init

                print("Creating env")
                env = MultiAgentVecEnv(make_env(breaker)) if TREATMENT_NAME.lower() == "multi" else DummyVecEnv([make_env(breaker)])
                print("Created env")

                if alg == "SAC":
                    model = SAC("MlpPolicy", env, verbose=2, learning_starts=10000)  
                elif alg == "PPO":
                    model = PPO("MlpPolicy", env, verbose=2, n_steps=256)

                print("Created model")
                print("Starting learning")
                model.learn(total_timesteps=250000)
                wandb.finish()
                print("Learning ended")
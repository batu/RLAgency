from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy
import os
import random

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper, UnityToMultiGymWrapper 

from rlnav.logging import WANDBMonitor


from pathlib import Path



breaker = random.randint(0,1000)


PROJECT_NAME = "Jump"
GROUP_NAME = "Default18"

base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROJECT_NAME}")
ENV_PATH = base_bath / r"18\Env.exe"  

config = {
    "ENV_PATH":ENV_PATH
}
def make_env(rank, seed=0):
    def _init():
        unity_env = UnityEnvironment(str(config["ENV_PATH"]), base_port=6172 + rank)
        # env = UnityToGymWrapper(unity_env)
        env = UnityToMultiGymWrapper(unity_env)
        env.seed(seed + rank)
        env = WANDBMonitor(env, config, project=PROJECT_NAME, group=GROUP_NAME)
        return env
    set_random_seed(seed)
    return _init

print("Creating env")
env = MultiAgentVecEnv(make_env(breaker))
print("Created env")

model = PPO("MlpPolicy", env, verbose=2, n_steps=512)  
print("Created model")
print("Starting learning")
model.learn(total_timesteps=250000, log_interval=None)
print("Learning ended")
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

base_bath = Path(r"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\Benchmark")
ENV_PATH = base_bath / r"Empty_1\Env.exe"  
ENV_PATH = base_bath / r"Empty_8\Env.exe"


breaker = random.randint(0,1000)
# ENV_PATH = base_bath / f"Basic_DF_4\\Env.exe"

# base_bath = Path(r"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\Debug")

# ENV_PATH = base_bath / r"Ball3D_12Agent_1Frame\UnityEnvironment.exe" #                             106 after 200000 221 sec        | 359 after 200000 237 (nsteps 1024)
config = {
    "ENV_PATH":ENV_PATH
}

log_dir = "results/Empty_8"
def make_env(rank, seed=0):
    def _init():
        unity_env = UnityEnvironment(str(config["ENV_PATH"]), base_port=6122 + rank)
        # env = UnityToGymWrapper(unity_env)
        env = UnityToMultiGymWrapper(unity_env)
        env.seed(seed + rank)
        # env = Monitor(env, log_dir)
        # env = WANDBMonitor(env, config, run_name)
        env = MonitorMulti(env, log_dir)
        return env
    set_random_seed(seed)
    return _init

print("Creating env")
num_cpu = 8  # Number of processes to use
# env = DummyVecEnv([make_env(i) for i in range(num_cpu)])
env = MultiAgentVecEnv(make_env(breaker))
print("Created env")

model = PPO("MlpPolicy", env, verbose=2, n_steps=512)  
print("Created model")
print("Starting learning")
model.learn(total_timesteps=100000)
print("Learning ended")
# reward, std = evaluate_policy(model, model.get_env().envs[0], n_eval_episodes=5)
# print(reward)
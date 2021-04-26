from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, MultiAgentVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor, MonitorMulti
from stable_baselines3.common.evaluation import evaluate_policy
import os

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper, UnityToMultiGymWrapper 

from pathlib import Path

base_bath = Path(r"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\Debug")
ENV_PATH = base_bath / r'\Ball3D_2Agent_1Frame\UnityEnvironment.exe' # 117 after 100000 215 sec
ENV_PATH = base_bath / r"Ball3D_1Agent_1Frame\UnityEnvironment.exe"  # 79 after 100000 317 sec  | 40.3 after 200000 439 (Vec x 12) | 110 after 200000 411 (nestep 1024)
ENV_PATH = base_bath / r"Ball3D_12Agent_1Frame\UnityEnvironment.exe" #                             106 after 200000 221 sec        | 359 after 200000 237 (nsteps 1024)

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

def make_env(rank, seed=0):
    def _init():
        unity_env = UnityEnvironment(str(ENV_PATH), base_port=6022 + rank)
        # env = UnityToGymWrapper(unity_env)
        env = UnityToMultiGymWrapper(unity_env)
        env.seed(seed + rank)
        # env = Monitor(env, log_dir)
        env = MonitorMulti(env, log_dir)
        return env
    set_random_seed(seed)
    return _init

print("Creating env")
num_cpu = 12  # Number of processes to use
# env = DummyVecEnv([make_env(i) for i in range(num_cpu)])
env = MultiAgentVecEnv(make_env(0), 12)
print("Created env")

model = PPO("MlpPolicy", env, verbose=2, n_steps=160)
print("Created model")
print("Starting learning")
model.learn(total_timesteps=100000)
print("Learning ended")
# reward, std = evaluate_policy(model, model.get_env().envs[0], n_eval_episodes=5)
# print(reward)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

ENV_PATH = r"C:\Users\user\Desktop\RLNav\NavigationEnvironments\P0\Jump93_Single\Env.exe"

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

def make_env(rank, seed=0):
    def _init():
        unity_env = UnityEnvironment(str(ENV_PATH), base_port=6022 + rank)
        env = UnityToGymWrapper(unity_env)
        env.seed(seed + rank)
        env = Monitor(env, log_dir)
        return env
    set_random_seed(seed)
    return _init

num_cpu = 4  # Number of processes to use
env = DummyVecEnv([make_env(i) for i in range(num_cpu)])

model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=100000 * 8)
reward, std = evaluate_policy(model, model.get_env().envs[0], n_eval_episodes=10)
print(reward)
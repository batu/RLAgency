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

# %%
from stable_baselines3 import PPO, SAC
import torch

model_name = "EasyBaseline1"
# loaded_model = PPO.load(r"C:\Users\batua\Desktop\RLNav\RLAgency\results\Jump\PPO_Occupancy\Occupancy93_8_LRcon-4_BS1024_GAE0.95_HU256\Jump_92.8%.zip")
# loaded_model = SAC.load(r"C:\Users\batua\Desktop\RLNav\Results\Chubb\SAC\Baseline\BestNetwork0.63.zip")
loaded_model = SAC.load(f"C:/Users/batua/Desktop/RLNav/Models/{model_name}.zip")

# %%
loaded_model.policy

# %%
class PPOOnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(PPOOnnxablePolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, input):
        action_hidden, value_hidden = self.extractor(input)
        return (self.action_net(action_hidden), self.value_net(value_hidden))

class SACOnnxablePolicy(torch.nn.Module):
    def __init__(self,  actor):
        super(SACOnnxablePolicy, self).__init__()
        
        # Removing the flatten layer because it can't be onnxed
        self.actor = torch.nn.Sequential(actor.latent_pi, actor.mu)

    def forward(self, input):
        return self.actor(input)

class SACOnnxablePolicyValue(torch.nn.Module):
    def __init__(self,  critic):
        super(SACOnnxablePolicyValue, self).__init__()
        
        # Removing the flatten layer because it can't be onnxed
        self.critic = critic.qf0

    def forward(self, input):
        return self.critic(input)

if isinstance(loaded_model, PPO):  
    new_model = PPOOnnxablePolicy(loaded_model.policy.mlp_extractor, loaded_model.policy.action_net, loaded_model.policy.value_net)
else:
    new_model = SACOnnxablePolicyValue(loaded_model.policy.critic)

dummy_input = torch.randn(1, 508)
loaded_model.policy.to("cpu")
torch.onnx.export(new_model, dummy_input, f"C:/Users/batua/Desktop/RLNav/Models/ONNX/{model_name}Value.onnx", opset_version=9)
print(new_model)
print(new_model(dummy_input))


if False:
    PROTOTYPE_NAME = "Urban"
    EXPERIMENT_NAME = f"Longrun"
    PROTOTYPE_PATH_NAME = "Urban"
    base_bath = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_PATH_NAME}")

    ENV_NAME = "EasyBaseline"
    ENV_PATH = base_bath / fr"{ENV_NAME}\Env.exe"  
    TREATMENT_NAME = f"LongLocal"
        
    with open(Path("rlnav/configs/SAC_rlnav_config.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb_config, network_config, alg_config, channels = setup_configurations(config)
    def make_env():
        
        def _init():
          unity_env = UnityEnvironment(str(ENV_PATH), base_port=5000 + random.randint(0,5000), side_channels=channels)
          env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
          env = WANDBMonitor(env, wandb_config, prototype=PROTOTYPE_NAME, experiment=EXPERIMENT_NAME, treatment=TREATMENT_NAME)
          return env
        return _init

    env = MultiAgentVecEnv(make_env())
    final_success_rate = test_model(env, loaded_model)
    print(final_success_rate)

# %%



# %%

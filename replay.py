
# %%
import random
from stable_baselines3 import PPO
import torch
from rlnav.logging import WANDBMonitor, test_model
from pathlib import Path
import random
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
import numpy as np

# %%
PROTOTYPE_NAME = "JUMP"
ENV_NAME = "Occupancy93_TS2_32"

env_path = Path(fr"C:\Users\batua\Desktop\RLNav\NavigationEnvironments\{PROTOTYPE_NAME}\{ENV_NAME}\Env.exe")
loaded_model = PPO.load(r"C:\Users\batua\Desktop\RLNav\RLAgency\results\Jump\PPO_TimeScale\Occupancy93_TS2_32_LRcon-4_BS1024_HU512\Jump_98.0%.zip")

unity_env = UnityEnvironment(str(env_path), base_port=5000 + random.randint(0,1000))
env = UnityToMultiGymWrapper(unity_env)

# %%
def test_model(env, model, test_count=1000, det=True):
  results = []
  obs = env.reset()
  remaining_eval_counts = [1000 // env.num_agents for _ in range(env.num_agents)]

  while any(remaining_eval_counts):
    obs = np.squeeze(np.array(obs))
    actions, _ = model.predict(obs, deterministic=det)
    obs, rews, dones, infos = env.step(actions)
    for idx, (done, rew) in enumerate(zip(dones, rews)):
      if done:
        results.append(rew)
        remaining_eval_counts[idx] = max(0, remaining_eval_counts[idx] - 1)
        # print(remaining_eval_counts)
  return np.mean(results) 

print(test_model(env, loaded_model, test_count=1000))

# class OnnxablePolicy(torch.nn.Module):
#   def __init__(self, extractor, action_net, value_net):
#     super(OnnxablePolicy, self).__init__()
#     self.extractor = extractor
#     self.action_net = action_net
#     self.value_net = value_net

#   def forward(self, input):
#     action_hidden, value_hidden = self.extractor(input)
#     return (self.action_net(action_hidden), self.value_net(value_hidden))

# new_model = OnnxablePolicy(loaded_model.policy.mlp_extractor, loaded_model.policy.action_net, loaded_model.policy.value_net)
# loaded_model.policy.to("cpu")
# # for _ in range(1000):
#   # print(new_model(torch.tensor(torch.randn(1, 93).clone().detach(), dtype=torch.float32)))


# dummy_input = torch.randn(1, 93)
# torch.onnx.export(new_model, dummy_input, "jump93_97%.onnx", opset_version=9)
# # 

# # %%

# %%
from stable_baselines3 import PPO, SAC
import torch
# loaded_model = PPO.load(r"C:\Users\batua\Desktop\RLNav\RLAgency\results\Jump\PPO_Occupancy\Occupancy93_8_LRcon-4_BS1024_GAE0.95_HU256\Jump_92.8%.zip")
loaded_model = SAC.load(r"C:\Users\batua\Desktop\RLNav\RLAgency\results\ExtraTrained2.zip")


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

if isinstance(loaded_model, PPO):  
    new_model = PPOOnnxablePolicy(loaded_model.policy.mlp_extractor, loaded_model.policy.action_net, loaded_model.policy.value_net)
else:
    new_model = SACOnnxablePolicy(loaded_model.policy.actor)

print(new_model)
# %%
dummy_input = torch.randn(1, 505)
loaded_model.policy.to("cpu")
torch.onnx.export(new_model, dummy_input, "Urban505_995.onnx", opset_version=9)


# %%

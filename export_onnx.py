# %%
from stable_baselines3 import PPO
import torch
loaded_model = PPO.load(r"C:\Users\user\Desktop\RLNav\RLAgency\Results\0_Pipeline\Mutliple\PPO\SB4.zip")


# %%
class OnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, input):
        action_hidden, value_hidden = self.extractor(input)
        return (self.action_net(action_hidden), self.value_net(value_hidden))

new_model = OnnxablePolicy(loaded_model.policy.mlp_extractor, loaded_model.policy.action_net, loaded_model.policy.value_net)
# %%
new_model(torch.tensor([[0,0,0,0,0,0]], dtype=torch.float32))


# %%
dummy_input = torch.randn(1, 6)
loaded_model.policy.to("cpu")
torch.onnx.export(new_model, dummy_input, "navmodel.onnx", opset_version=9)


# %%

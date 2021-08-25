from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import torch as th
import platform
from pathlib import Path
import yaml
import wandb

activations = {"relu": th.nn.ReLU,
               "tanh": th.nn.Tanh,
               "relu6": th.nn.ReLU6,
               "leakyrelu": th.nn.LeakyReLU,
               "selu":th.nn.SELU}





def setup_configurations(config):

    wandb.finish()
    wandb_config = {
        "Algorithm": "SAC",
        "Source":"LapLocaltop" if platform.system() == "Windows" else "Collab", 
    }
    network_config = configure_network(config, wandb_config)
    alg_config     = configure_algorithm(config, wandb_config)
    environment_channel, engine_channel = configure_unity(config, wandb_config)
    
    wandb_config.update(alg_config)
    wandb_config.update(network_config)

    return wandb_config, network_config, alg_config, [environment_channel, engine_channel]


def configure_network(config, wandb_config):
    neural_network_config = config["network_config"]
    wandb_config["Activation"] = neural_network_config["activation_fn"]
    wandb_config["VF"]         = neural_network_config["net_arch"]["qf"]
    wandb_config["PI"]         = neural_network_config["net_arch"]["pi"]
    neural_network_config["activation_fn"] = activations[neural_network_config["activation_fn"].lower()]
    return neural_network_config

def configure_algorithm(config, wandb_config):
    # handle learning rate schedule here.
    alg_config = config["sac_config"]
    return alg_config
    
def configure_unity(config, wandb_config) -> EnvironmentParametersChannel:
    observation_config = config["observation_config"]
    environment_channel = EnvironmentParametersChannel()
    for key, value in observation_config.items():
        environment_channel.set_float_parameter(key, value)

    env_config = config["environment_config"]
    for key, value in env_config.items():
        environment_channel.set_float_parameter(key, value)
        
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale = env_config["time_scale"])
    
    
    wandb_config.update(observation_config)
    wandb_config.update(env_config)

    return environment_channel, engine_channel

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import MultiAgentVecEnv

from mlagents_envs.exception import UnityTimeOutException, UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import torch as th

import os, yaml
from torch._C import device
import random
import time
import datetime
import numpy as np
import torch as th
from torch import nn

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
import wandb
import json

from rlnav.scene_graphs import URBAN_SCENE_GRAPH_JSONSTR

from pathlib import Path

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GraphConv

from copy import deepcopy

SLICE_DICT =  {
    "CanSeeGoal": slice(0,1),
    "Direction" : slice(1,4),
    "DirectionNormalized": slice(4,7),
    "MagnitudeNormalized": slice(7,8),
    "RemainingJumpCount" : slice(8,9),
    "Velocity": slice(9,12),
    "IsGrounded": slice(12,13),
    "AgentPosition": slice(13,16),
    "GoalPosition": slice(16,19),
    "AgentGoalPosition": slice(13,19),
}


URBAN_SCENE_GRAPH = json.loads(URBAN_SCENE_GRAPH_JSONSTR)


class AggregateGCN(nn.Module):
    def __init__(self, in_dim, num_layers, h_feats=32, embedding_size=16):
        super(AggregateGCN, self).__init__()
        self.three_layers = num_layers == 3
        self.conv1 = GraphConv(in_dim, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)
        self.embedding = nn.Linear(h_feats, embedding_size)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        if self.three_layers:
            h = F.relu(self.conv3(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.embedding(hg)
        
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        graph_hidden_layers_count:int =3,
        graph_hidden_layers_width:int =32,
        graph_embedding_size:int =32,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super(CustomNetwork, self).__init__()
        
        self.GNN = AggregateGCN(6, graph_hidden_layers_count, graph_hidden_layers_width, graph_embedding_size)
        self.GNNEmbedding_dim = self.GNN.embedding.out_features

        g = dgl.graph((URBAN_SCENE_GRAPH["SourceNodes"], URBAN_SCENE_GRAPH["DestinationNodes"]), num_nodes=URBAN_SCENE_GRAPH["NumNodes"], device = th.device("cuda:0"))
        self.g = dgl.add_self_loop(g)
        self.g.to(th.device('cuda:0'))

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(26 + self.GNNEmbedding_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(26 + self.GNNEmbedding_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
            
        """
        graph_batch = self.get_GNN_batch(features["graph"])
        graph_embedding = self.GNN(graph_batch, graph_batch.ndata["feat"])

        combined_features = th.cat([features["vector"][:, :26], graph_embedding], dim=1)
        return self.policy_net(combined_features), self.value_net(combined_features)

    def get_GNN_batch(self, features_batch):
        graphs = []
        for features in features_batch:
            g = deepcopy(self.g)
            g.to(th.device('cuda:0'))
            g.ndata["feat"] = features
            graphs.append(g)
        return dgl.batch(graphs)
    
class GraphActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        self.graph_hidden_layers_count = net_arch[0]["graph_hidden_layers_count"]
        self.graph_hidden_layers_width = net_arch[0]["graph_hidden_layers_width"]
        self.graph_embedding_size= net_arch[0]["graph_embedding_size"]
        super(GraphActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


        # Disable orthogonal initialization
        self.ortho_init = True

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.graph_hidden_layers_count, self.graph_hidden_layers_width, self.graph_embedding_size)
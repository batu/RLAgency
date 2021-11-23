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
import pdb 

from copy import copy

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
import wandb
import json

from rlnav.scene_graphs import AGENT_GOAL_SCENE_GRAPH_JSONSTR, URBAN_SCENE_GRAPH_JSONSTR

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


# SCENE_GRAPH = json.loads(URBAN_SCENE_GRAPH_JSONSTR)
SCENE_GRAPH = json.loads(AGENT_GOAL_SCENE_GRAPH_JSONSTR)


class SingleLayerGCN(nn.Module):
    def __init__(self, in_dim, latent_feat=16, embedding_size=8, local_steps=2, global_steps=4, agg_style="node"):
        super(SingleLayerGCN, self).__init__()
        self.latent_feat = latent_feat
        self.linear = nn.Linear(in_dim, latent_feat)
        self.one_conv = GraphConv(latent_feat, latent_feat, allow_zero_in_degree=True)
        self.embedding = nn.Linear(latent_feat, embedding_size)

        self.local_steps = local_steps
        self.aggregation_style = agg_style
        self.device = "cuda:0"

        self.residual = agg_style == "residual"


    def get_agent_nodes(self, g, agent_index=0):
        """
        node_count:int the number of nodes in each of the graph. We assume this is the same.
        example agent_subgraph_idxs if we assume agent_index=0
        [ 0,  2,  4,  6,  8, 10, 12, 14] 
        https://docs.dgl.ai/generated/dgl.batch.html
        """
        batch_num_nodes = set(g.batch_num_nodes())
        assert len(batch_num_nodes) != 1, "The graphs in the batch have different number of nodes."
        
        node_count = batch_num_nodes.pop()
        agent_subgraph_idxs = th.arange(agent_index, node_count * g.batch_size, step=node_count, device=self.device)

        batched_agent_nodes = dgl.node_subgraph(g, agent_subgraph_idxs)
        return batched_agent_nodes

    def forward(self, g, x):
 
        # Apply graph convolution and activation.
        x = F.relu(self.linear(x))
        g.ndata["hidden"] = x
        agent_nodes = self.get_agent_nodes(g, 0)
        
        for _ in range(self.local_steps):
            x = x + F.relu(self.one_conv(g, x)) if self.residual else F.relu(self.one_conv(g, x))

        x = agent_nodes.ndata["hidden"]  
        return self.embedding(x)



class AggregateGCN(nn.Module):
    def __init__(self, in_dim, latent_feat=16, embedding_size=8, local_steps=2, global_steps=4, agg_style="node"):
        super(AggregateGCN, self).__init__()
        self.latent_feat = latent_feat
        self.linear = nn.Linear(in_dim, latent_feat)
        self.convs = nn.ModuleList([GraphConv(latent_feat, latent_feat, allow_zero_in_degree=True) 
                                    for _ in range(local_steps)])
        self.embedding = nn.Linear(latent_feat, embedding_size)

        self.aggregation_style = agg_style
        self.residual = agg_style == "residual"
        self.device = "cuda:0"


    def get_agent_nodes(self, g, agent_index=0):
        """
        node_count:int the number of nodes in each of the graph. We assume this is the same.
        example agent_subgraph_idxs if we assume agent_index=0
        [ 0,  2,  4,  6,  8, 10, 12, 14] 
        https://docs.dgl.ai/generated/dgl.batch.html
        """
        batch_num_nodes = set(g.batch_num_nodes())
        assert len(batch_num_nodes) != 1, "The graphs in the batch have different number of nodes."
        
        node_count = batch_num_nodes.pop()
        agent_subgraph_idxs = th.arange(agent_index, node_count * g.batch_size, step=node_count, device=self.device)

        batched_agent_nodes = dgl.node_subgraph(g, agent_subgraph_idxs)
        return batched_agent_nodes

    def forward(self, g, x):
 
        # Apply graph convolution and activation.
        x = F.relu(self.linear(x))
        for layer in self.convs:
            x = x + F.relu(layer(g, x)) if self.residual else F.relu(layer(g, x))
        
        g.ndata["hidden"] = x
        agent_nodes = self.get_agent_nodes(g, 0)
        x = agent_nodes.ndata["hidden"]  
        return self.embedding(x)

class AdamGCN(nn.Module):
    def __init__(self, in_dim, latent_feat=32, embedding_size=16, local_steps=2, global_steps=4, agg_style="node"):
        super(AdamGCN, self).__init__()
        
        self.latent_feat = latent_feat
        self.f = nn.Linear(in_dim, latent_feat)
        self.g = GraphConv(latent_feat, latent_feat)
        self.h = GraphConv(latent_feat, latent_feat)
        self.device = "cuda:0"

        self.aggregation_style = agg_style

        self.local_steps = local_steps
        self.global_steps = global_steps

        self.embedding = nn.Linear(latent_feat, embedding_size)

    def forward(self, g, x):
        node_count = g.batch_num_nodes()[0].item()

        original_x = x
        x = F.relu(self.f(x))
        # https://docs.dgl.ai/en/0.6.x/api/python/nn.pytorch.html#globalattentionpooling

        for _ in range(self.local_steps):
            x += F.relu(self.g(g, x))

        for _ in range(self.global_steps):
            x += F.relu(self.h(g, x))

        if self.aggregation_style == "node":
            x = x.reshape((-1, node_count, self.latent_feat)) # find a dgl way of doing.
            x = np.squeeze(x[:, 0, :])
            return F.relu(self.embedding(x))

        if self.aggregation_style == "attention":
            player_attention = original_x[:, 1] == 0 # 1 is the index of typeID and 0 is the player type ID 
            # !!! Batu look into why the shape has changed.
            g.ndata['h'] = x * player_attention[:, None]
            # Calculate graph representation by max readout.
            hg = dgl.mean_nodes(g, "h")
            return F.relu(self.embedding(hg))

        elif self.aggregation_style == "max":
            with g.local_scope():
                g.ndata['h'] = x
                # Calculate graph representation by max readout.
                hg = dgl.max_nodes(g, 'h')
                return F.relu(self.embedding(hg))

        elif self.aggregation_style == "mean":
            with g.local_scope():
                g.ndata['h'] = x
                # Calculate graph representation by average readout.
                hg = dgl.mean_nodes(g, 'h')
                return F.relu(self.embedding(hg))
        else:
            raise "The Aggregation mode is wrong!"


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
        network_type, 
        graph_feature_size=28,
        graph_hidden_layers_width:int=256,
        graph_embedding_size:int=32,
        local_steps=2,
        global_steps=4,
        agg_style="node",
        edge_style="bi_directional",
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super(CustomNetwork, self).__init__()
        
        self.vector_size = 26
        
        self.node_feature_size = graph_feature_size
        self.vector_size = 3

        self.GNN = network_type(self.node_feature_size, latent_feat=graph_hidden_layers_width, embedding_size=graph_embedding_size, local_steps=local_steps, global_steps=global_steps, agg_style=agg_style)
        self.GNNEmbedding_dim = self.GNN.embedding.out_features

        g = dgl.graph((SCENE_GRAPH["SourceNodes"], SCENE_GRAPH["DestinationNodes"]), num_nodes=SCENE_GRAPH["NumNodes"], device = th.device("cuda:0"))

        # Edge styles --> 
            # Other way around, child to parent
            # Current parent to child
            # bidirectional.
            # self loops
        if edge_style == "bi_directional":
            self.g = dgl.add_reverse_edges(g)
        elif edge_style == "parent_to_child":
            pass
        elif edge_style == "child_to_parent":
            self.g = dgl.reverse(g)

        self.g = dgl.add_self_loop(g)
        self.g.to(th.device('cuda:0'))

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.vector_size + self.GNNEmbedding_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.vector_size + self.GNNEmbedding_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
            
        """
        graph_batch = self.get_GNN_batch(features["graph"])
        graph_embedding = self.GNN(graph_batch, graph_batch.ndata["feat"])

        combined_features = th.cat([features["vector"][:, 9:12], graph_embedding], dim=1)
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

        self.graph_feature_size = observation_space["graph"].shape[-1]

        self.network_type = net_arch[0]["network_type"] if "network_type" in net_arch[0] else AdamGCN
        
        self.graph_hidden_layers_width = net_arch[0]["graph_hidden_layers_width"] if "graph_hidden_layers_width" in net_arch[0] else 256
        self.graph_embedding_size= net_arch[0]["graph_embedding_size"] if "graph_embedding_size" in net_arch[0] else 32

        self.agg_style = net_arch[0]["agg_style"] if "agg_style" in net_arch[0] else "node"
        self.edge_style = net_arch[0]["edge_style"]  if "edge_style" in net_arch[0] else "bi_directional"

        self.local_steps = net_arch[0]["local_steps"] if "local_steps" in net_arch[0] else 2
        self.global_steps = net_arch[0]["global_steps"] if "global_steps" in net_arch[0] else 3


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
        self.mlp_extractor = CustomNetwork(network_type=self.network_type,
                                           graph_feature_size=self.graph_feature_size,
                                           graph_hidden_layers_width=self.graph_hidden_layers_width,
                                           graph_embedding_size=self.graph_embedding_size,
                                           local_steps=self.local_steps,
                                           global_steps=self.global_steps,
                                           agg_style=self.agg_style,
                                           edge_style=self.edge_style
        )
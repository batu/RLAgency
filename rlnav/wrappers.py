import itertools
from dgl.convert import graph
import numpy as np
from typing import Any, Dict, List, Tuple, Union

import gym
from gym import error, spaces
import torch as th

import dgl
from copy import deepcopy

import json

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

from rlnav.scene_graphs import URBAN_SCENE_GRAPH_JSONSTR
URBAN_SCENE_GRAPH = json.loads(URBAN_SCENE_GRAPH_JSONSTR)

class GraphDictWrapper(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment

    """

    def __init__(
        self,
        env: gym.Env,
        normalize:bool=True,
    ):
        super(GraphDictWrapper, self).__init__(env=env)

        g = dgl.graph((URBAN_SCENE_GRAPH["SourceNodes"], URBAN_SCENE_GRAPH["DestinationNodes"]), num_nodes=URBAN_SCENE_GRAPH["NumNodes"], device = th.device("cuda:0"))
        self.g = dgl.add_self_loop(g)
        self.g.to(th.device('cuda:0'))

        scene_graph_features = np.array(URBAN_SCENE_GRAPH["Features"])      
        self.scene_graph_selected_features = np.concatenate([scene_graph_features[:,1:4], scene_graph_features[:,1:4]], axis=1)

        if normalize:
            self.scene_graph_selected_features = np.concatenate([scene_graph_features[:,1:4], scene_graph_features[:,1:4]], axis=1) / 141.0
        self.vector_size = env.observation_space.shape[0]

        self.observation_space = gym.spaces.Dict(
            spaces={
                "vector": gym.spaces.Box(-1, 1, (self.vector_size,)),
                "graph": gym.spaces.Box(-1, 1, (364, 6),)
            }
        )

    def preprocess_graph(self, features, reset=False):
        if reset:
            graph_features = np.array([obs[0][SLICE_DICT["AgentGoalPosition"]] - self.scene_graph_selected_features for obs in features])
        else:
            graph_features = np.array([obs[SLICE_DICT["AgentGoalPosition"]] - self.scene_graph_selected_features for obs in features])
        return graph_features

    
    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        observations = self.env.reset(**kwargs)
        graph_features = self.preprocess_graph(observations, reset=True)
        obs_dict = np.array([{
            "vector":v[0],
            "graph":g,
        } for v, g in zip(observations, graph_features)])

        return obs_dict

    def step(self, action: Union[np.ndarray, int]):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observations, rewards, dones, infos = self.env.step(action)
        graph_features = self.preprocess_graph(observations)
        obs_dict = np.array([{
            "vector":v,
            "graph":g,
        } for v, g in zip(observations, graph_features)])

        return obs_dict, rewards, dones, infos


    def close(self) -> None:
        """
        Closes the environment
        """
        super(ConvDictWrapper, self).close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times



class ConvDictWrapper(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment

    """


    def __init__(
        self,
        env: gym.Env,
    ):
        super(ConvDictWrapper, self).__init__(env=env)

        self.vector_size = 19 + 26 + 6  # This includes the whiskers
        self.depthmap_size = (1, 7, 7)
        self.occupancy_size = (9, 5, 9)

        vector_size = 19 + 26 + 6  # This includes the whiskers
        depth_size = 55 - 6
        occupancy_size = 405

        self.vector_end = vector_size
        self.depthmask_end = self.vector_end + depth_size
        self.occupancy_end = self.depthmask_end + occupancy_size
 

        self.observation_space = gym.spaces.Dict(
            spaces={
                "vector": gym.spaces.Box(-1, 1, (self.vector_size,)),
                "depthmap": gym.spaces.Box(0, 1, self.depthmap_size),
                "occupancy": gym.spaces.Box(0, 1, self.occupancy_size),
            }
        )

    def preprocess_np(self, input_list):

        input_array = np.array(input_list).squeeze()
        vector_obs = input_array[:, :self.vector_end]
        depth_obs = input_array[:,  self.vector_end:self.depthmask_end]
        occupancy_obs = input_array[:, self.depthmask_end: self.occupancy_end]     

        depth_obs2d = depth_obs.reshape(-1, *self.depthmap_size)
        occupancy_obs3d = occupancy_obs.reshape(-1, *self.occupancy_size)

        return vector_obs, depth_obs2d, occupancy_obs3d

    
    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        observations = self.env.reset(**kwargs)
        vector_obs, depth_obs2d, occupancy_obs3d = self.preprocess_np(observations)
        obs_dict = np.array([{
            "vector":v,
            "depthmap":d,
            "occupancy": o
        } for v,d,o in zip(vector_obs, depth_obs2d, occupancy_obs3d)])

        return obs_dict

    def step(self, action: Union[np.ndarray, int]):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observations, rewards, dones, infos = self.env.step(action)
        vector_obs, depth_obs2d, occupancy_obs3d = self.preprocess_np(observations)
        obs_dict = np.array([{
            "vector":v,
            "depthmap":d,
            "occupancy": o
        } for v,d,o in zip(vector_obs, depth_obs2d, occupancy_obs3d)])

        return obs_dict, rewards, dones, infos


    def close(self) -> None:
        """
        Closes the environment
        """
        super(ConvDictWrapper, self).close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times

from dataclasses import dataclass

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

from rlnav.scene_graphs import URBAN_SCENE_GRAPH_JSONSTR, AGENT_GOAL_SCENE_GRAPH_JSONSTR
# SCENE_GRAPH = json.loads(URBAN_SCENE_GRAPH_JSONSTR)
SCENE_GRAPH = json.loads(AGENT_GOAL_SCENE_GRAPH_JSONSTR)


@dataclass
class AllGraphNodes:
    """Class for keeping track of features of a Graph Features."""
    gameobjectIDs: np.array
    typeIDs: np.array
        
    rel_agent_poss: np.array
    rel_goal_poss: np.array
    
    abs_poss: np.array
    abs_rots: np.array
    abs_scales: np.array

    loc_poss: np.array
    loc_rots: np.array
    loc_scales: np.array
        
    rel_agent_poss: np.array
    rel_goal_poss: np.array
        
    def update_rel_goal_pos(self, goal_pos:np.array):
        self.rel_goal_poss = goal_pos - self.abs_poss
    
    def update_rel_agent_pos(self, agent_pos:np.array):
        self.rel_agent_poss = agent_pos - self.abs_poss
    
    def get_updated_features(self, agent_goal_pos):
        agent_pos = agent_goal_pos[:3]
        goal_pos  = agent_goal_pos[3:]
        self.update_rel_goal_pos(goal_pos)
        self.update_rel_agent_pos(agent_pos)
        return np.concatenate([self.static_features, self.rel_agent_poss, self.rel_goal_poss], axis=1)

    def get_updated_features_nostatic(self, agent_goal_pos):
        agent_pos = agent_goal_pos[:3]
        goal_pos  = agent_goal_pos[3:]
        self.update_rel_goal_pos(goal_pos)
        self.update_rel_agent_pos(agent_pos)
        return np.concatenate([self.rel_agent_poss, self.rel_goal_poss], axis=1)

    def get_updated_features_onlypos(self, agent_goal_pos):
        graph_features = np.zeros((2,3))
        graph_features[0] = agent_goal_pos[:3]
        graph_features[1] = agent_goal_pos[3:]
        return graph_features

    def get_updated_features_batch(self, agent_goal_pos):
        agent_pos = agent_goal_pos[:, :, :3]
        goal_pos  = agent_goal_pos[:, :, 3:]
        self.update_rel_goal_pos(goal_pos)
        self.update_rel_agent_pos(agent_pos)
        return np.concatenate([self.batched_static_features, self.rel_agent_poss, self.rel_goal_poss], axis=2)

    def normalize(self):
        def normalize(v):
            return v / (np.abs(v).max() + 0.00001)
        
        self.gameobjectIDs = normalize(self.gameobjectIDs)
        self.typeIDs = normalize(self.typeIDs)

        self.abs_poss = normalize(self.abs_poss) 
        self.abs_rots = normalize(self.abs_rots)
        self.abs_scales = normalize(self.abs_scales)

        self.loc_poss = normalize(self.loc_poss)
        self.loc_rots = normalize(self.loc_rots)
        self.loc_scales = normalize(self.loc_scales)

        self.rel_agent_poss = normalize(self.rel_agent_poss)
        self.rel_goal_poss = normalize(self.rel_goal_poss)
        

    def JsonToAllGraphNodes(JSON_list:list, normalize=True):
        JSON_array = np.array(JSON_list)
        num_nodes = JSON_array.shape[0]

        agent_features = np.array(JSON_list[0])
        goal_features  = np.array(JSON_list[1])

        gameobjectIDs = JSON_array[:,0]
        typeIDs = JSON_array[:,1]

        abs_poss = np.array(JSON_array[:,2:5])
        abs_rots = np.array(JSON_array[:,5:9])
        abs_scales = np.array(JSON_array[:,9:12])

        loc_poss = np.array(JSON_array[:,12:15])
        loc_rots = np.array(JSON_array[:,15:19])
        loc_scales = np.array(JSON_array[:,19:22])

        agent_pos = agent_features[2:5]
        goal_pos = goal_features[2:5]

        rel_agent_poss = agent_pos - abs_poss  
        rel_goal_poss  = goal_pos - abs_poss

        all_nodes= AllGraphNodes(gameobjectIDs=gameobjectIDs,
                         typeIDs=typeIDs,

                         abs_poss=abs_poss,
                         abs_rots=abs_rots,
                         abs_scales=abs_scales,

                         loc_poss=loc_poss,
                         loc_rots=loc_rots,
                         loc_scales=loc_scales, 

                         rel_agent_poss=rel_agent_poss,
                         rel_goal_poss=rel_goal_poss)
        
        if normalize:
            all_nodes.normalize()

        static_features_len = 22
        batch_size = 256

        all_nodes.static_features = JSON_array[:,0: static_features_len]
        all_nodes.batched_static_features = np.repeat(all_nodes.static_features, batch_size, axis=1).reshape(-1, num_nodes, static_features_len)
        
        return all_nodes


class AbsPosOnlyWrapper(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment

    """

    def __init__(
        self,
        env: gym.Env,
        normalize:bool=True,
    ):
        super(AbsPosOnlyWrapper, self).__init__(env=env)
        self.observation_space = spaces.Box(-1, 1, dtype=np.float32, shape=(6,))

    
    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        observations = self.env.reset(**kwargs)
        
        obs = np.array(observations)[:,:,13:19] * 10
        return obs

    def step(self, action: Union[np.ndarray, int]):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observations, rewards, dones, infos = self.env.step(action)
        obs = np.array(observations)[:,13:19] * 10

        return obs, rewards, dones, infos


    def close(self) -> None:
        """
        Closes the environment
        """
        super(AbsPosOnlyWrapper, self).close()

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


class AbsPosVecOnlyWrapper(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment

    """

    def __init__(
        self,
        env: gym.Env,
        normalize:bool=True,
    ):
        super(AbsPosVecOnlyWrapper, self).__init__(env=env)

    
    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        observations = self.env.reset(**kwargs)
        
        obs = np.concatenate([np.array(observations)[:,:,9:12], np.array(observations)[:,:,13:19]], axis=2)
        return obs

    def step(self, action: Union[np.ndarray, int]):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observations, rewards, dones, infos = self.env.step(action)
        obs = np.concatenate([np.array(observations)[:,9:12], np.array(observations)[:,13:19]], axis=1)

        return obs, rewards, dones, infos


    def close(self) -> None:
        """
        Closes the environment
        """
        super(AbsPosVecOnlyWrapper, self).close()

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

        self.all_graph_nodes = AllGraphNodes.JsonToAllGraphNodes(SCENE_GRAPH["Features"], normalize=normalize)      

        self.vector_size = env.observation_space.shape[0]

        self.observation_space = gym.spaces.Dict(
            spaces={
                "vector": gym.spaces.Box(-1, 1, (self.vector_size,)),
                "graph": gym.spaces.Box(-1, 1, self.get_graph_feature_size())
            }
        )

    def get_graph_feature_size(self):
        obs = self.env.reset()
        graph_features = self.preprocess_graph(obs)
        return graph_features.shape[1:]

    def preprocess_graph(self, features, reset=False):
        
        features = np.squeeze(features)
        # data = np.repeat(features, 366, axis=1).reshape(-1, 366, 505)
        # graph_features = self.all_graph_nodes.get_updated_features_batch(data[:, : , 13:19])
        graph_features = np.array([self.all_graph_nodes.get_updated_features_onlypos(obs[SLICE_DICT["AgentGoalPosition"]]) for obs in features])
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
        super(GraphDictWrapper, self).close()

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

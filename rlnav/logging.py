import wandb
import gym
import numpy as np
import time

import csv
import json

from typing import List, Optional, Tuple, Union
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class WANDBMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        config: dict,
        run_name: str
    ):
        super(WANDBMonitor, self).__init__(env=env)
        self.t_start = time.time()
        wandb.init(project=run_name, config=config)
        num_agents = len(env.reset())

        filename = run_name

        
        filename = filename + "." + "csv"
        self.file_handler = open(filename, "wt")
        self.file_handler.write("#%s\n" % json.dumps({"t_start": self.t_start, "env_id": env.spec and env.spec.id}))
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t"))
        self.logger.writeheader()
        self.file_handler.flush()


        self.rewards = [[]] * num_agents
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []

        self.total_steps = 0
        self.episide_count = 0

        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observations, rewards, dones, infos = self.env.step(action)
        for idx, stepreturn in enumerate(zip(observations, rewards, dones, infos)):
            observation, reward, done, info = stepreturn
            self.rewards[idx].append(reward)
            if done:
                ep_rew = sum(self.rewards[idx])
                ep_len = len(self.rewards[idx])
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)
                self.episode_times.append(time.time() - self.t_start)

                self.episide_count += 1
                wandb.log({"episode_reward":ep_rew,
                          "episode_len":ep_len,
                          "episode_time":ep_info["t"]},
                          step=self.total_steps)
                if self.logger:
                    self.logger.writerow(ep_info)
                    self.file_handler.flush()
                self.rewards[idx].clear()
            self.total_steps += 1

        return observations, rewards, dones, infos

    def close(self) -> None:
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

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

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

    
    rewards = [[]] 
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    printer_acquired = False
    total_steps = 0
    episode_count = 0


    def __init__(
        self,
        env: gym.Env,
        config: dict,
        project: str,
        group:str=None
    ):
        super(WANDBMonitor, self).__init__(env=env)
        self.t_start = time.time()
        wandb.init(project=project, group=group, config=config)

        self.run_name = project + " " + group
        obs = env.reset()
        self.num_agents =  1 if np.array(obs).ndim == 1 else len(obs)  
        WANDBMonitor.rewards *= self.num_agents

        self.log_frequency = 10000
        self.next_log_timestep = 0
        self.to_log = True

        self.rewards = [[]] * self.num_agents
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []

        self.max_obs = -np.inf
        self.min_obs = np.inf

        self.log_start_idx = 0

        self.printer = False
        if not WANDBMonitor.printer_acquired:
            WANDBMonitor.printer_acquired = True
            self.printer = True

        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    
    def log_to_console(self):
        if not self.printer: return

        time_elapsed = (time.time() - self.t_start)
        fps = int(self.total_steps / time_elapsed)
        ep_rews = WANDBMonitor.episode_rewards[self.log_start_idx:]
        ep_lens = WANDBMonitor.episode_lengths[self.log_start_idx:]

        eps_rew_mean = self.safe(np.mean, ep_rews)
        eps_len_mean = self.safe(np.mean, ep_lens)
        
        eps_rew_std = self.safe(np.std, ep_rews)
        eps_len_std = self.safe(np.std, ep_lens)

        best_ep  = self.safe(np.max, ep_rews)
        worst_ep = self.safe(np.min, ep_rews)

        print()
        print(self.run_name.center(70, "-"))
        print("|", "|".rjust(68))
        print("| Reward Mean".ljust(30), "|", f"{eps_rew_mean:.3f}  ±  {eps_rew_std:.3f}".ljust(35), "|")
        print("| EpLen  Mean".ljust(30), "|", f"{eps_len_mean:.3f}  ±  {eps_len_std:.3f}".ljust(35), "|")
        print("| Best and Worst Ep".ljust(30), "|", f"{best_ep:.3f}  |  {worst_ep:.3f}".ljust(35), "|")
        print("|", "|".rjust(68))
        print("| Biggest and Smallest Ob".ljust(30), "|", f"{self.max_obs:.3f} | {self.min_obs:.3f}".ljust(35), "|")
        print("|", "|".rjust(68))
        print("| FPS".ljust(30), "|", f"{fps}".ljust(35), "|")
        print("| Time".ljust(30), "|", f"{int(time_elapsed)}".ljust(35), "|")
        print("| Episodes".ljust(30), "|", f"{WANDBMonitor.episode_count}".ljust(35), "|")
        print("| Timesteps".ljust(30), "|", f"{WANDBMonitor.total_steps}".ljust(35), "|")
        print("".center(70, "-"))

        self.log_start_idx = WANDBMonitor.episode_count
        self.to_log = False
    
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
        if self.num_agents == 1:
            observations, rewards, dones, infos = [observations], [rewards], [dones], [infos]

        for idx, stepreturn in enumerate(zip(observations, rewards, dones, infos)):
            observation, reward, done, info = stepreturn

            self.max_obs = max(np.max(observation), self.max_obs)
            self.min_obs = min(np.min(observation), self.min_obs)

            self.rewards[idx].append(reward)
            if done:
                ep_rew = sum(self.rewards[idx])
                ep_len = len(self.rewards[idx])
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                WANDBMonitor.episode_rewards.append(ep_rew)
                WANDBMonitor.episode_lengths.append(ep_len)
                self.episode_times.append(time.time() - self.t_start)

                WANDBMonitor.episode_count += 1
                wandb.log({"episode_reward":ep_rew,
                          "episode_len":ep_len,
                          "episode_time":ep_info["t"]},
                          step=WANDBMonitor.total_steps)
                self.rewards[idx].clear()
            WANDBMonitor.total_steps += 1
            if WANDBMonitor.total_steps > self.next_log_timestep:
                self.to_log = True
                self.next_log_timestep += self.log_frequency
        
        if self.to_log:
            self.log_to_console()

        if self.num_agents == 1:
            observations, rewards, dones, infos = observations[0], rewards[0], dones[0], infos[0]

        return observations, rewards, dones, infos

    def safe(self, fn, arr):
        return np.nan if len(arr) == 0 else fn(arr)

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

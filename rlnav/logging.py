from numpy.core.fromnumeric import mean
import wandb
import gym
import numpy as np
import time

import csv
import json
import glob

from typing import List, Optional, Tuple, Union
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from collections import defaultdict, deque
import os
from pathlib import Path


def test_model(env, model, test_count=1000, det=True):
  env_channel = env.envs[0].env_channel
  env_channel.set_float_parameter("testing", 1)

  results = []
  obs = env.reset()
  while len(results) < test_count:
    actions, _ = model.predict(obs, deterministic=det)
    obs, rews, dones, infos = env.step(actions)
    for (done, reward) in zip(dones, rews):
      if done:
        if reward == 1.0:
            results.append(1.0)
        elif reward == -1.0 or reward == 0.0:
            results.append(0.0)
        else:
            print(f"Final step reward is different than 1.0 or 0.0. Success calculations are wrong! The reward is: {reward}")
  env_channel.set_float_parameter("testing", 0)
  return np.mean(results)  

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
    
    test_results = []
    successes = deque([0 for _ in range(250)], 250)
    episode_rewards = deque([], 50)
    episode_lengths = deque([], 50)
    episode_times = deque([], 50)
    printer_acquired = False
    total_steps   = 0
    episode_count = 0
    WANDB_logger = None
    max_success_rate = 0
    dirpath = ""


    def __init__(
        self,
        env: gym.Env,
        config: dict,
        prototype: str,
        experiment:str=None,
        treatment:str=None,
    ):
        super(WANDBMonitor, self).__init__(env=env)
        self.reset_static_variables()

        self.env_channel = env.env_channel
        print(env)

        dirpath = Path(f"Results/{prototype}/{experiment}/{treatment}")
        WANDBMonitor.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)

        self.t_start = time.time()
        self.run = wandb.init(project=prototype, group=experiment, name=treatment, config=config)

        self.run_name = prototype + " " + experiment + " " + treatment
        obs = env.reset()
        self.num_agents =  1 if np.array(obs).ndim == 1 else len(obs)  

        self.log_frequency = 10000
        self.test_frequency = 250000
        self.next_log_timestep = 0
        self.next_test_timestep = 0
        
        self.test_count = 250
        self.testing = False
        self.to_log = True

        self.name_to_value = defaultdict(list)
        self.rewards = [list() for _ in range(self.num_agents)]
        self.actions = deque([], 1000)

        self.max_obs = -np.inf
        self.min_obs = np.inf

        self.max_action = -np.inf
        self.min_action = np.inf

        self.log_start_idx = 0

        self.printer = False
        if not WANDBMonitor.printer_acquired:
            WANDBMonitor.printer_acquired = True
            self.printer = True

        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()
        WANDBMonitor.WANDB_logger = self
    
    def record(self, key: str, value, exclude= None) -> None:
        self.name_to_value[key].append(value) 
        wandb.log({key:value},step=WANDBMonitor.total_steps)

    def log_to_console(self):
        if not self.printer: return

        mean_success_rate = self.safe(np.mean, WANDBMonitor.successes)
        wandb.log({"_MeanTrainingSuccessRate":mean_success_rate},
                          step=WANDBMonitor.total_steps)


        actions = np.array(self.actions)
        self.max_action = max(self.max_action, np.max(actions))
        self.min_action = min(self.min_action, np.min(actions))
        mean_actions, std_actions = np.mean(actions, axis=0),  np.std(actions, axis=0)

        time_elapsed = (time.time() - self.t_start)
        fps = int(self.total_steps / time_elapsed)
        ep_rews = WANDBMonitor.episode_rewards
        ep_lens = WANDBMonitor.episode_lengths

        eps_rew_mean = self.safe(np.mean, ep_rews)
        eps_len_mean = self.safe(np.mean, ep_lens)
        
        eps_rew_std = self.safe(np.std, ep_rews)
        eps_len_std = self.safe(np.std, ep_lens)

        best_ep  = self.safe(np.max, ep_rews)
        worst_ep = self.safe(np.min, ep_rews)

        headings = defaultdict(list)
        for key, value in self.name_to_value.items():
            heading = "NoHeading"
            if "/" in key:
                heading, name = key.split("/")
            headings[heading].append((name, self.safe(np.mean, value)))
        self.name_to_value.clear()

        print()
        print(self.run_name.center(70, "-"))
        print("|", "|".rjust(68))
        print("| Last 250 Success Rate".ljust(30), "|", f"{mean_success_rate:.1%}", "|", f"Max:{WANDBMonitor.max_success_rate:.3%}".ljust(35), "|")
        print("| Reward Mean".ljust(30), "|", f"{eps_rew_mean:.3f}  ±  {eps_rew_std:.3f}".ljust(35), "|")
        print("| EpLen  Mean".ljust(30), "|", f"{eps_len_mean:.3f}  ±  {eps_len_std:.3f}".ljust(35), "|")
        print("| Best and Worst Ep".ljust(30), "|", f"{best_ep:.3f}  |  {worst_ep:.3f}".ljust(35), "|")
        print("|", "|".rjust(68))
        print("| Biggest and Smallest Ob".ljust(30), "|", f"{self.max_obs:.3f} ({self.max_obs_idx}) | {self.min_obs:.3f} ({self.min_obs_idx})".ljust(35), "|")
        print("| Biggest and Smallest Act".ljust(30), "|", f"{self.max_action:.3f} | {self.min_action:.3f}".ljust(35), "|")
        with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
            print("| Mean Act".ljust(30), "|", f"{mean_actions}".ljust(35), "|")
            print("| Std Act".ljust(30), "|", f"{std_actions}".ljust(35), "|")
        for heading in headings:
            print("|", "|".rjust(68))
            for name, value in headings[heading]:
                 print(f"| {name}".ljust(30), "|", f"{value:.3f}".ljust(35), "|")
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
        for act in action:
            self.actions.append(act)

        if self.num_agents == 1:
            observations, rewards, dones, infos = [observations], [rewards], [dones], [infos]

        for idx, stepreturn in enumerate(zip(observations, rewards, dones, infos)):
            observation, reward, done, info = stepreturn

            step_max_obs = np.max(observation)
            if  step_max_obs > self.max_obs:
                self.max_obs = step_max_obs
                self.max_obs_idx = np.argmax(observation)

            step_min_obs = np.min(observation)
            if step_min_obs < self.min_obs:
                self.min_obs = step_min_obs
                self.min_obs_idx = np.argmin(observation)
            

            self.rewards[idx].append(reward)
            if done:
                
                if self.testing:
                    if reward == 1.0:
                        WANDBMonitor.test_results.append(1.0)
                    elif reward == -1.0 or reward == 0.0:
                        WANDBMonitor.test_results.append(0.0)

                    if len(WANDBMonitor.test_results) > self.test_count:
                        self.testing = False
                        mean_test_success_rate = self.safe(np.mean, WANDBMonitor.test_results)
                        wandb.log({"_SuccessRate":mean_test_success_rate},
                                                step=WANDBMonitor.total_steps)
                        self.env_channel.set_float_parameter("testing", 0)
                else:
                    if reward == 1.0:
                        self.successes.append(1.0)
                    elif reward == -1.0 or reward == 0.0:
                        self.successes.append(0.0)
                    else:
                        print(f"Final step reward is different than 1.0 or 0.0. Success calculations are wrong! The reward is: {reward}")
                        

                    

                ep_rew = sum(self.rewards[idx])
                ep_len = len(self.rewards[idx])
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                WANDBMonitor.episode_rewards.append(ep_rew)
                WANDBMonitor.episode_lengths.append(ep_len)
                WANDBMonitor.episode_times.append(time.time() - self.t_start)

                WANDBMonitor.episode_count += 1
                wandb.log({"episode_reward":ep_rew,
                          "episode_len":ep_len,
                          "episode_time":ep_info["t"],
                          "episode_success":reward},
                          step=WANDBMonitor.total_steps)
                self.rewards[idx].clear()

            
            WANDBMonitor.total_steps += 1
            if WANDBMonitor.total_steps > self.next_log_timestep and not self.testing:
                self.to_log = True
                self.next_log_timestep += self.log_frequency

            if WANDBMonitor.total_steps > self.next_test_timestep:
                self.testing = True
                WANDBMonitor.test_results.clear()
                self.env_channel.set_float_parameter("testing", 1)
                self.next_test_timestep += self.test_frequency
        
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
        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/test.pt.trace.json")[0], "trace.pt.trace.json")
        self.run.log_artifact(profile_art)

        super(WANDBMonitor, self).close()
        self.reset_static_variables()

    def reset_static_variables(self):
        WANDBMonitor.test_results = []
        WANDBMonitor.successes = deque([0 for _ in range(250)], 250)
        WANDBMonitor.episode_rewards = deque([], 50)
        WANDBMonitor.episode_lengths = deque([], 50)
        WANDBMonitor.episode_times = deque([], 50)
        WANDBMonitor.printer_acquired = False
        WANDBMonitor.total_steps   = 0
        WANDBMonitor.episode_count = 0
        WANDBMonitor.WANDB_logger = None
        WANDBMonitor.max_success_rate = 0

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

import gym
import time
import numpy as np
import d4rl
import torch

class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def kitchen_env_reset(env, base_observation: torch.Tensor):
    observation = env.reset()
    observation, obs_goal = observation[:30], observation[30:]
    
    # Random dataset state for proprioceptive states
    obs_goal[:9] = base_observation[:9].cpu()

    return observation, obs_goal


def kitchen_env_step(env, action):
    next_observation, reward, done, info = env.step(action)
    next_observation = next_observation[:30]
    return next_observation, reward, done, info
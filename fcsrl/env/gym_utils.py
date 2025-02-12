from typing import Any
import gymnasium as gym
import mujoco
import numpy as np
import time

from safety_gymnasium.bases.underlying import VisionEnvConf
import random


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_s, r, terminate, trunc, info = self.env.step(action)

        if ('goal_met' in info) and info['goal_met']:
            info['terminate'] = True
        else:
            info['terminate'] = False
        return next_s, r, terminate, trunc, info


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_repeat: int):
        super().__init__(env)
        self.n_repeat = n_repeat

    def step(self, action):
        total_r = 0.0
        total_c = 0.0
        total_term = total_trunc = total_psuedo_term = False

        total_info = {}
        for _ in range(self.n_repeat):
            next_s, r, terminate, trunc, info = self.env.step(action)
            total_r += r
            total_c += info['cost']
            total_term = (total_term or terminate)
            total_trunc = (total_trunc or trunc)
            total_psuedo_term = (total_psuedo_term or info.get("terminate", False))

            if terminate or trunc:
                break

        total_info["cost"] = total_c
        total_info["terminate"] = total_psuedo_term

        return next_s, total_r, total_term, total_trunc, total_info


class VisionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1.0, (64, 64, 3))

    def observation(self, observation):
        VisionEnvConf.vision_size = (64, 64)
        obs = self.env.task._obs_vision() / 255
        return obs


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Randomize environment parameters on each reset, for robust training.
    You can extend this with custom logic that modifies friction,
    hazard locations, sensor noise, etc.
    """

    def __init__(self, env, friction_range=(0.8, 1.2), noise_range=(0.0, 0.1)):
        super().__init__(env)
        self.friction_range = friction_range
        self.noise_range = noise_range

    def reset(self, **kwargs):
        # 1) Randomize friction
        new_friction = random.uniform(*self.friction_range)
        self.set_friction(new_friction)

        # 2) Randomize sensor noise
        new_noise = random.uniform(*self.noise_range)
        self.set_sensor_noise(new_noise)

        # Now call the underlying env reset
        obs, info = super().reset(**kwargs)
        return obs, info

    def set_friction(self, friction_val):
        """
        Pseudocode: If your underlying environment or
        safety_gymnasium environment has an attribute for friction,
        set it here. E.g.:
            self.env.unwrapped.model.geom_friction[...] = friction_val
        or call self.env.task.set_friction(friction_val)
        Depending on how safety_gymnasium organizes friction parameters.
        """
        model = self.env.unwrapped.task.agent.engine.model
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == "agent":
                self.env.unwrapped.task.agent.engine.model.geom_friction[i] *= friction_val

    def set_sensor_noise(self, noise_val):
        """
        Example: self.env.unwrapped.model.sensor_noise = noise_val
        Or if there's a built-in function to set noise, call that.
        """
        self.env.unwrapped.task.agent.engine.model.sensor_noise = noise_val
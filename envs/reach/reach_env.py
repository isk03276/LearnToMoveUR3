import random
import math

from envs.base_env import BaseEnv
from utils.geometry import get_distance_between_two_pts

import numpy as np
from pyrep.objects.shape import Shape
from gym.spaces import Box


class ReachEnv(BaseEnv):
    """
    Reach env class.
    """

    def __init__(
        self,
        scene_file="envs/reach/reach.ttt",
        use_arm_camera: bool = False,
        rendering: bool = True,
    ):
        super().__init__(scene_file, use_arm_camera, rendering)
        self.target = Shape("TargetPoint")
        self.target_x_range = (-0.35, 0.35)
        self.target_y_range = (-0.35, 0.35)
        self.target_z_range = (0.05, 0.5)
        self.min_distance_from_base = 0.2

        self.reach_threshold = 0.05

    def _define_observation_space(self) -> Box:
        """
        Define/Get observation space.
        """
        observation_space = Box(float("-inf"), float("inf"), (17,))
        return observation_space

    def get_obs(self) -> np.ndarray:
        """
        Get agent's observation.
        The observation contains robot state and a relative position of target object
        Returns:
            (np.ndarray) : agent's observation
        """
        obs = self.get_robot_state()
        target_realtive_position = self.get_object_position_relative_to_base_link(
            self.target
        )
        obs.extend(target_realtive_position)
        return np.array(obs)

    def get_reward(self) -> float:
        """
        This reward function is based on the distance between target object and tip.
        Returns:
            (float) : reward
        """
        distance_between_tip_and_target = self.get_distance_from_tip(
            self.target.get_position()
        )
        return -math.log10(distance_between_tip_and_target / 10 + 1)

    def reset_objects(self):
        """
        Reset a target object.
        """
        def get_random_point():
            random_point_x = random.uniform(self.target_x_range[0], self.target_x_range[1])
            random_point_y = random.uniform(self.target_y_range[0], self.target_y_range[1])
            random_point_z = random.uniform(self.target_z_range[0], self.target_z_range[1])
            return np.array([random_point_x, random_point_y, random_point_z])
        distance = float("-inf")
        while distance < self.min_distance_from_base:
            random_point = get_random_point()
            distance = get_distance_between_two_pts(np.zeros(2), random_point[:2])
        self.target.set_position(list(random_point))

    def is_goal_state(self) -> bool:
        """
        If the target object and the tip are close, it is considered as a goal state.
        """
        distance_between_tip_and_target = self.get_distance_from_tip(
            self.target.get_position()
        )
        return distance_between_tip_and_target <= self.reach_threshold

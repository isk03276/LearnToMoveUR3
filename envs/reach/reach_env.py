import random
import math

from envs.base_env import BaseEnv
from utils.geometry import get_distance_between_two_pts

from pyrep.objects.shape import Shape


class ReachEnv(BaseEnv):
    def __init__(
        self,
        scene_file="envs/reach/reach.ttt",
        use_arm_camera: bool = False,
        rendering: bool = True,
    ):
        super().__init__(scene_file, use_arm_camera, rendering)
        self.target = Shape("TargetPoint")
        self.target_x_range = (0.2, 0.4)
        self.target_y_range = (-0.2, 0.2)
        self.target_z_range = (0.4, 0.7)

        self.reach_threshold = 0.01

    def get_reward(self):
        distance_between_tip_and_target = self._get_distance_between_tip_and_target()
        if distance_between_tip_and_target <= self.reach_threshold:
            return 1
        return -math.log10(distance_between_tip_and_target / 100 + 1)

    def get_done(self):
        return super().get_done()

    def reset_objects(self):
        self._reset_target()

    def _reset_target(self):
        random_point_x = random.uniform(self.target_x_range[0], self.target_x_range[1])
        random_point_y = random.uniform(self.target_y_range[0], self.target_y_range[1])
        random_point_z = random.uniform(self.target_z_range[0], self.target_z_range[1])
        self.target.set_position([random_point_x, random_point_y, random_point_z])

    def _get_distance_between_tip_and_target(self):
        return get_distance_between_two_pts(
            self.target.get_position(), self.tip.get_position()
        )

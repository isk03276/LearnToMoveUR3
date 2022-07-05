import random
import math

from envs.base_env import BaseEnv

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
        distance_between_tip_and_target = self.get_distance_from_tip(
            self.target.get_position()
        )
        return -math.log10(distance_between_tip_and_target / 100 + 1)

    def reset_objects(self):
        random_point_x = random.uniform(self.target_x_range[0], self.target_x_range[1])
        random_point_y = random.uniform(self.target_y_range[0], self.target_y_range[1])
        random_point_z = random.uniform(self.target_z_range[0], self.target_z_range[1])
        self.target.set_position([random_point_x, random_point_y, random_point_z])

    def is_goal_state(self):
        distance_between_tip_and_target = self.get_distance_from_tip(
            self.target.get_position()
        )
        return distance_between_tip_and_target <= self.reach_threshold

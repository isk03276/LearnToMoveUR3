import random

from envs.base_env import BaseEnv
from utils.geometry import get_distance_between_two_pts

from pyrep.objects.shape import Shape


class ReachEnv(BaseEnv):
    def __init__(self, scene_file="envs/reach/reach.ttt", rendering=True):
        super().__init__(scene_file, rendering)
        self.target = Shape("TargetPoint")
        self.target_x_range = (0.1, 0.3)
        self.target_y_range = (-0.2, 0.2)
        self.target_z_range = (0.2, 0.4)
        
        self.reach_threshold = 0.01
        
    def get_reward(self):
        return 1 if self._get_distance_between_tip_and_target() <= self.reach_threshold else 0
    
    def get_done(self):
        done = super().get_done()
        if done:
            return done
        return True if self._get_distance_between_tip_and_target() <= self.reach_threshold else False
        
    def reset_objects(self):
        self._reset_target()
    
    def _reset_target(self):
        random_point_x = random.uniform(self.target_x_range[0], self.target_x_range[1])
        random_point_y = random.uniform(self.target_y_range[0], self.target_y_range[1])
        random_point_z = random.uniform(self.target_z_range[0], self.target_z_range[1])
        self.target.set_position([random_point_x, random_point_y, random_point_z])
        
    def _get_distance_between_tip_and_target(self):
        return get_distance_between_two_pts(self.target.get_position(), self.tip.get_position())
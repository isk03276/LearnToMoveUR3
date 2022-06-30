import random

from envs.base_env import BaseEnv

from pyrep.objects.shape import Shape


class ReachEnv(BaseEnv):
    def __init__(self, scene_file="envs/reach/reach.ttt", rendering=True):
        super().__init__(scene_file, rendering)
        self.target = Shape("TargetPoint")
        self.target_x_range = (0.1, 0.3)
        self.target_y_range = (-0.2, 0.2)
        self.target_z_range = (0.2, 0.4)
        self.max_time_step = 300
        self.reach_threshold = 0.01
        
    def get_reward(self):
        return 1
    
    def get_done(self):
        return False
        
    def reset_objects(self):
        self._reset_target()
    
    def _reset_target(self):
        random_point_x = random.uniform(self.target_x_range[0], self.target_x_range[1])
        random_point_y = random.uniform(self.target_y_range[0], self.target_y_range[1])
        random_point_z = random.uniform(self.target_z_range[0], self.target_z_range[1])
        self.target.set_position([random_point_x, random_point_y, random_point_z])
from utils.image import resize_image

import gym
from gym.spaces import Box
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.ur3 import UR3
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper
from pyrep.objects.vision_sensor import VisionSensor


class BaseEnv(gym.Env):
    def __init__(self, scene_file:str, rendering=True):
        super().__init__()
        self.env = PyRep()
        self.env.launch(scene_file, headless=not rendering)
        self.env.start()
        self.arm = None
        self.gripper = None
        self.tip = None
        self.camera = VisionSensor("kinect_rgb")
        self.gripper_velocity = 0.2
        self._init_robot()
        self.observation_space = Box(0, 255, (84, 84, 3))
        self.action_space = Box(-3.14, 3.14, (7,))
        self.max_time_step = 300
        self.current_time_step = 0
    
    def _init_robot(self):
        self.arm = UR3()
        self.gripper = Robotiq85Gripper()
        self.tip = self.arm.get_tip()
        
    def get_obs(self):
        original_obs = self.camera.capture_rgb()
        width = self.observation_space.shape[0]
        height = self.observation_space.shape[1]
        obs = resize_image(original_obs, width, height)
        return obs
    
    def get_reward(self):
        raise NotImplementedError
    
    def reset_objects(self):
        raise NotImplementedError
    
    def get_done(self):
        return True if self.current_time_step >= self.max_time_step else False
    
    def reset(self):
        self.env.stop()
        self.env.start()
        self.reset_objects()
        self.env.step()
        return self.get_obs()
    
    def get_info(self):
        return {}
    
    def step(self, action):
        self.current_time_step += 1
        arm_control = action[:-1]
        gripper_control = action[-1]
        gripper_control = 1.0 if action[-1] > 0.5 else 0.0
        self.arm.set_joint_target_positions(arm_control)
        self.gripper.actuate(gripper_control, self.gripper_velocity)
        self.env.step()
        return self.get_obs(), self.get_reward(), self.get_done(), self.get_info()
        
    def close(self):
        self.env.stop()
        self.env.shutdown()
    
    
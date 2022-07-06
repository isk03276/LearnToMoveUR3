from utils.image import resize_image
from utils.geometry import get_distance_between_two_pts

import gym
from gym.spaces import Box
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.ur3 import UR3
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper
from pyrep.objects.vision_sensor import VisionSensor


class BaseEnv(gym.Env):
    def __init__(
        self, scene_file: str, use_arm_camera: bool = False, rendering: bool = True
    ):
        super().__init__()
        self.use_arm_camera = use_arm_camera
        self.env = PyRep()
        self.env.launch(scene_file, headless=not rendering)
        self.env.start()
        self.arm = None
        self.gripper = None
        self.tip = None
        self.third_view_camera = VisionSensor("kinect_rgb")
        self.arm_camera = VisionSensor("arm_camera_rgb")
        self.gripper_velocity = 0.2
        self._init_robot()

        self.max_time_step = 300
        self.current_time_step = 0

        # for the agent to keep the goal states
        self.max_consecutive_visit_to_goal = 5
        self.cur_consecutive_visit_to_goal = 0

        channel_num = 6 if use_arm_camera else 3
        self.observation_space = Box(0, 255, (84, 84, channel_num))
        self.action_space = Box(-1.0, 1, (7,))

    def _init_robot(self):
        self.arm = UR3()
        self.gripper = Robotiq85Gripper()
        self.tip = self.arm.get_tip()

    def get_obs(self):
        width = self.observation_space.shape[0]
        height = self.observation_space.shape[1]
        obs = self.third_view_camera.capture_rgb()
        if self.use_arm_camera:
            first_view_image = self.arm_camera.capture_rgb()
            obs = np.concatenate((obs, first_view_image), axis=2)
        obs = resize_image(obs, width, height)
        return obs

    def get_reward(self):
        raise NotImplementedError

    def reset_objects(self):
        raise NotImplementedError

    def is_goal_state(self):
        raise NotImplementedError

    def get_done_and_info(self):
        is_success = self.is_success()
        return self.time_over() | is_success, {"success": is_success}

    def is_success(self):
        if self.is_goal_state():
            self.cur_consecutive_visit_to_goal += 1
        else:
            self.cur_consecutive_visit_to_goal = 0
        return self.cur_consecutive_visit_to_goal >= self.max_consecutive_visit_to_goal

    def time_over(self):
        return True if self.current_time_step >= self.max_time_step else False

    def reset(self):
        self.current_time_step = 0
        self.cur_consecutive_visit_to_goal = 0
        self.env.stop()
        self.env.start()
        self.arm.set_control_loop_enabled(False)
        self.arm.set_motor_locked_at_zero_velocity(True)
        self.reset_objects()
        self.env.step()
        return self.get_obs()

    def step(self, action):
        self.current_time_step += 1
        arm_control = action[:-1]
        gripper_control = action[-1]
        gripper_control = 1.0 if action[-1] > 0.0 else 0.0
        self.arm.set_joint_target_velocities(arm_control)
        self.gripper.actuate(gripper_control, self.gripper_velocity)
        self.env.step()
        done, info = self.get_done_and_info()
        return self.get_obs(), self.get_reward(), done, info

    def close(self):
        self.env.stop()
        self.env.shutdown()

    def get_distance_from_tip(self, object_position):
        return get_distance_between_two_pts(self.tip.get_position(), object_position)

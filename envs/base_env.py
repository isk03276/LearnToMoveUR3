from typing import Tuple, List

from utils.image import resize_image
from utils.geometry import get_distance_between_two_pts

import gym
from gym.spaces import Box
import numpy as np
from pyrep import PyRep
from pyrep.robots.arms.ur3 import UR3
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.objects import Object


class BaseEnv(gym.Env):
    """
    Base environment class for learning behavior of UR3 robot.
    """

    def __init__(
        self, scene_file: str, use_arm_camera: bool = False, rendering: bool = True
    ):
        super().__init__()
        self.use_arm_camera = use_arm_camera
        self.env = PyRep()
        self.env.launch(scene_file, headless=not rendering)
        self.env.start()
        self.arm = None
        self.arm_base_link = None
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

        self.observation_space = self._define_observation_space()
        self.action_space = Box(-1.0, 1, (7,))

    def _init_robot(self):
        """
        Initialize robot.
        Assume that (robot arm) UR3 with (gripper) Robotiq85Gripper.
        """
        self.arm = UR3()
        self.arm_base_link = Shape("UR3_link2")
        self.gripper = Robotiq85Gripper()
        self.tip = self.arm.get_tip()

    def _define_observation_space(self) -> Box:
        """
        Define/Get observation space.
        Returns:
            (Box) : defined observation space
        """
        raise NotImplementedError

    def get_obs(self) -> np.ndarray:
        """
        Get agent's observation.
        Returns:
            (np.ndarray) : agent's observation
        """
        raise NotImplementedError

    def get_reward(self) -> float:
        """
        Get reward for state/action.
        We currently assume reward for state.
        Returns:
            (float) : reward
        """
        raise NotImplementedError

    def reset_objects(self):
        """
        Reset objects in env. (ex. target_object)
        """
        raise NotImplementedError

    def is_goal_state(self) -> bool:
        """
        Whether the current state is a goal state or not.
        Returns :
            (bool) : if current state is a goal state, then return True
        """
        raise NotImplementedError

    def render(self) -> np.ndarray:
        """
        Get image observation from cameras.
        Returns:
            np.ndarray: image observation
        """
        width = self.observation_space.shape[0]
        height = self.observation_space.shape[1]
        obs = self.third_view_camera.capture_rgb()
        if self.use_arm_camera:
            first_view_image = self.arm_camera.capture_rgb()
            obs = np.concatenate((obs, first_view_image), axis=2)
        obs = resize_image(obs, width, height)
        return obs

    def get_done_and_info(self) -> Tuple[bool, dict]:
        """
        Get done and info.
        The done indicates whether the episode is over.
        The info contains several informations such as whether the episode is successed.
        Returns :
            (bool) : done
            (dict) : info
        """
        is_success = self.is_success()
        return self.time_over() | is_success, {"success": is_success}

    def is_success(self) -> bool:
        """
        Whether the episode is successed.
        Returns:
            (bool) : if the episode is successed, then return True
        """
        if self.is_goal_state():
            self.cur_consecutive_visit_to_goal += 1
        else:
            self.cur_consecutive_visit_to_goal = 0
        return self.cur_consecutive_visit_to_goal >= self.max_consecutive_visit_to_goal

    def time_over(self) -> bool:
        """
        Whether the episode is progressing beyond the set maximum time step. 
        Returns:
            (bool) : if timestep is overed, then return True
        """
        return True if self.current_time_step >= self.max_time_step else False

    def reset(self) -> np.ndarray:
        """
        Reset env and Start new episode.
        Returns:
            (np.ndarray) : initial obs
        """
        self.current_time_step = 0
        self.cur_consecutive_visit_to_goal = 0
        self.env.stop()
        self.env.start()
        self.arm.set_control_loop_enabled(False)
        self.arm.set_motor_locked_at_zero_velocity(True)
        self.reset_objects()
        self.env.step()
        return self.get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute the action and Observe the new obs, reward, etc.
        Args:
            action (np.ndarray): the action decided by the agent

        Returns:
            (np.ndarray) : new obs
            (float) : reward
            (bool) : done
            (dict) : info
        """
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
        """
        Close the env.
        """
        self.env.stop()
        self.env.shutdown()

    def get_distance_from_tip(self, object_position: np.ndarray) -> float:
        """
        Get distance between tip and target object
        Args:
            object_position (np.ndarray): position of the target object

        Returns:
            (float): distance
        """
        return get_distance_between_two_pts(self.tip.get_position(), object_position)

    def get_robot_state(self) -> List[float]:
        """
        Get the robot state.
        The robot state contains the robot arm state and the gripper state.
        The robot arm state contains positions of arm joints and velocities of arm joints.
        The gripper state contains mean of open amount of gripper and mean of velocities of gripper joints.
        Returns:
            (list) : robot state
        """
        arm_joint_positions = self.arm.get_joint_positions()
        arm_joint_velocities = self.arm.get_joint_velocities()
        gripper_positions = self.gripper.get_open_amount()
        gripper_velocities = self.gripper.get_joint_velocities()
        gripper_position_mean = sum(gripper_positions) / len(gripper_positions)
        gripper_velocity_mean = sum(gripper_velocities) / len(gripper_velocities)
        robot_state = arm_joint_positions
        robot_state.extend(arm_joint_velocities)
        robot_state.append(gripper_position_mean)
        robot_state.append(gripper_velocity_mean)
        return robot_state

    def get_object_position_relative_to_base_link(
        self, target_object: Object
    ) -> np.ndarray:
        """
        Get relative positoin of the target object to robot arm base link.
        Args:
            target_object (Object): target object

        Returns:
            (np.ndarray): position array containing x, y, z
        """
        return target_object.get_position(relative_to=self.arm_base_link)

from sim2real.sensing.robot_sensing_manager import RobotSensingManager
from sim2real.sensing.object_sensing_manager import ObjectSensingManager
from sim2real.interfaces.base_interface import BaseInterface

import numpy as np


class ReachInterface(BaseInterface):
    def __init__(
        self,
        arm_state_topic: str,
        gripper_state_topic: str,
        object_state_topic: str,
        arm_control_topic: str,
        gripper_control_topic: str,
    ):
        self.arm_state_topic = arm_state_topic
        self.gripper_state_topic = gripper_state_topic
        self.object_state_topic = object_state_topic
        self.arm_control_topic = arm_control_topic
        self.gripper_control_topic = gripper_control_topic
        
        super().__init__()
        

        self.arm_control_manager = None
        self.gripper_control_manager = None

    def init_sensing_managers(self):
        self.arm_sensing_manager = RobotSensingManager(self.arm_state_topic)
        self.gripper_sensing_manager = RobotSensingManager(self.gripper_state_topic)
        self.object_sensing_manager = ObjectSensingManager(self.object_state_topic)
        self.sensing_managers = [
            self.arm_sensing_manager,
            self.gripper_sensing_manager,
            self.object_sensing_manager,
        ]

    def do_sensing(self):
        recent_common_seq = self.get_recent_common_seq()
        arm_joint_state = self.arm_sensing_manager.do_sensing(recent_common_seq)
        gripper_joint_state = self.gripper_sensing_manager.do_sensing(recent_common_seq)
        object_state = self.object_sensing_manager.do_sensing(recent_common_seq)
        state = self.make_state(arm_joint_state, gripper_joint_state, object_state)
        return state

    def do_control(self, action: np.ndarray):
        pass

    def make_state(self, arm_joint_state, gripper_joint_state, object_state):
        state = []
        arm_joint_positions = arm_joint_state["joint_positions"]
        arm_joint_velocities = arm_joint_state["joint_velocities"]
        gripper_joint_positions = gripper_joint_state["joint_positions"]
        gripper_joint_positions_mean = sum(gripper_joint_positions) / len(
            gripper_joint_positions
        )
        gripper_joint_velocities = gripper_joint_state["joint_velocities"]
        gripper_joint_velocities_mean = sum(gripper_joint_velocities) / len(
            gripper_joint_velocities
        )
        object_positions = object_state["positions"]

        state.extend(arm_joint_positions)
        state.append(gripper_joint_positions_mean)
        state.extend(arm_joint_velocities)
        state.append(gripper_joint_velocities_mean)
        state.extend(object_positions)

        return state

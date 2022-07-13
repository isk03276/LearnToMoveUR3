from sim2real.sensing.robot_sensing_manager import RobotSensingManager
from sim2real.sensing.object_sensing_manager import ObjectSensingManager
from sim2real.interfaces.base_interface import BaseInterface

import numpy as np


class ReachInterface(BaseInterface):
    def __init__(self,
                 arm_state_topic: str,
                 gripper_state_topic: str,
                 object_state_topic: str,
                 arm_control_topic: str,
                 gripper_control_topic: str):
        super().__init__()
        self.arm_sensing_manager = RobotSensingManager(arm_state_topic)
        self.gripper_sensing_manager = RobotSensingManager(gripper_state_topic)
        self.object_sensing_manager = ObjectSensingManager(object_state_topic)
        self.arm_control_manager = None
        self.gripper_control_manager = None
    
    def do_sensing(self):
        state = []
        arm_joint_state = self.arm_sensing_manager.do_sensing()
        gripper_joint_state = self.gripper_sensing_manager.do_sensing()
        object_state = self.object_sensing_manager.do_sensing()
        state.extend(arm_joint_state[:6])
        state.append(sum(gripper_joint_state[:2]) / sum(gripper_joint_state[:2]))
        state.extend(arm_joint_state[6:])
        state.append(sum(gripper_joint_state[2:]) / sum(gripper_joint_state[2:]))
        state.extend(object_state)
        return state
    
    def do_control(self, action: np.ndarray):
        pass
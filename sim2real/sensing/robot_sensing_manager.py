from sim2real.sensing.base_sensing_manager import BaseSensingManager

import rospy
from sensor_msgs.msg import JointState


class RobotSensingManager(BaseSensingManager):
    def __init__(self, topic_name: str):
        super().__init__(topic_name)
        
    def _init_ros(self):
        rospy.Subscriber(self.topic_name, JointState, self._sensing_callback)

    def _sensing_callback(self, msg: JointState):
        sensored_data = []
        joint_positions = msg.position
        joint_velocities = msg.velocity
        sensored_data.extend(joint_positions)
        sensored_data.extend(joint_velocities)
        self.sensored_data = sensored_data
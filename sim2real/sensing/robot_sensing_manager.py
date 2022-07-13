from sim2real.sensing.base_sensing_manager import BaseSensingManager

import rospy
from sensor_msgs.msg import JointState


class RobotSensingManager(BaseSensingManager):
    """
    Manager class for sensing the robot joint state.
    Sensored data : 1-D list containing joint positions and velocities.

    Args:
        BaseSensingManager (_type_): _description_
    """

    def __init__(self, topic_name: str):
        super().__init__(topic_name)

    def _init_ros(self):
        rospy.Subscriber(
            self.topic_name, JointState, self._sensing_callback, queue_size=10
        )

    def _sensing_callback(self, msg: JointState):
        self.seqs.append(msg.header.seq)
        sensored_data = dict()
        sensored_data["joint_positions"] = msg.position
        sensored_data["joint_velocities"] = msg.velocity
        self.sensored_data_list.append(sensored_data)

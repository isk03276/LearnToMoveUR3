from sim2real.sensing.base_sensing_manager import BaseSensingManager
from utils.ros import pose_msg_to_point_quat_list

import rospy
from geometry_msgs.msg import PoseArray


class ObjectSensingManager(BaseSensingManager):
    """
    Sensing manager class for sensing coordinates of objects.
    Sensored data : coordinates list of objects. (1-D)
    """

    def __init__(self, topic_name: str):
        super().__init__(topic_name)

    def _init_ros(self):
        rospy.Subscriber(
            self.topic_name, PoseArray, self._sensing_callback, queue_size=10
        )

    def _sensing_callback(self, msg: PoseArray):
        self.seqs.append(msg.header.seq)
        sensored_data = dict()
        point_list = []
        for pose in msg.poses:
            point, _ = pose_msg_to_point_quat_list(pose)
            point_list.extend(point)
        sensored_data["positions"] = point_list
        self.sensored_data_list.append(sensored_data)

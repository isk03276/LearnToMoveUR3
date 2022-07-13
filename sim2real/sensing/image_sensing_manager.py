from sim2real.sensing.base_sensing_manager import BaseSensingManager

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageSensingManager(BaseSensingManager):
    def __init__(self, topic_name: str):
        super().__init__(topic_name)
        self.bridge = CvBridge()

    def _init_ros(self):
        rospy.Subscriber(self.topic_name, Image, self._sensing_callback, queue_size=10)

    def _sensing_callback(self, msg: Image):
        self.seqs.append(msg.header.seq)
        sensored_data = dict()
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        sensored_data["image"] = img
        self.sensored_data_list.append(sensored_data)

from sim2real.sensing.base_sensing_manager import BaseSensingManager

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageSensingManager(BaseSensingManager):
    def __init__(self, topic_name: str):
        super().__init__(topic_name)
        self.bridge = CvBridge()
        
    def _init_ros(self):
        rospy.Subscriber(self.topic_name, Image, self._sensing_callback)
        
    def _sensing_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.sensored_data = self.convert_imgmsg_to_img(img)
        
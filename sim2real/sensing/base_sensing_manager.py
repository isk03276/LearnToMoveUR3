from abc import ABC, abstractmethod

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy

class BaseSensingManager(ABC):
    def __init__(self, topic_name: str):
        self.topic_name = topic_name
        self.sensored_data = None
        self._init_ros()

    @abstractmethod
    def _init_ros(self):
        raise NotImplementedError

    @abstractmethod
    def _sensing_callback(self, msg):
        raise NotImplementedError

    def do_sensing(self):
        if self.sensored_data is None:
            return None
        sensored_data = self.sensored_data
        self.sensored_data = None
        return sensored_data

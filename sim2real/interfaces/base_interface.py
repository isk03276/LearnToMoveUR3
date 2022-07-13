from abc import ABC, abstractmethod

import rospy


class BaseInterface(ABC):
    def __init__(self):
        self._init_ros()
        
    def _init_ros(self):
        rospy.init_node("sim_to_real_interface", anonymous=True)
    
    @abstractmethod
    def do_sensing(self, *args):
        raise NotImplementedError
    
    @abstractmethod
    def do_control(self, *args):
        raise NotADirectoryError
    
    
    
    
    
        
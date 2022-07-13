from abc import ABC, abstractmethod

import rospy


class BaseInterface(ABC):
    def __init__(self):
        self._init_ros()
        self.sensing_managers = []
        self.init_sensing_managers()

    def _init_ros(self):
        rospy.init_node("sim_to_real_interface", anonymous=True)

    @abstractmethod
    def do_sensing(self, *args):
        raise NotImplementedError

    def make_state(self, *args):
        raise NotImplementedError

    @abstractmethod
    def do_control(self, *args):
        raise NotImplementedError

    @abstractmethod
    def init_sensing_managers(self):
        raise NotImplementedError

    def get_recent_common_seq(self) -> int:
        common_seqs = []
        while not common_seqs:
            seqs_of_managers = [manager.get_seqs() for manager in self.sensing_managers]
            common_seqs = list(set.intersection(*map(set, seqs_of_managers)))
        recent_common_seq = max(common_seqs)
        return recent_common_seq

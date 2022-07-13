from abc import ABC, abstractmethod
from typing import List


class BaseSensingManager(ABC):
    """
    Sensing manager base class.
    """

    def __init__(self, topic_name: str):
        self.topic_name = topic_name
        self._init_sensored_data()
        self._init_ros()

    @abstractmethod
    def _init_ros(self):
        """
        Initialize ros topic subscriber.
        """
        raise NotImplementedError

    @abstractmethod
    def _sensing_callback(self, msg):
        """
        Process sensored data
        """
        raise NotImplementedError

    def do_sensing(self, seq: int = None) -> dict:
        """
        Get the sensored data
        """
        idx = self.seqs.index(seq)
        return self.sensored_data_list[idx]

    def get_seqs(self) -> List[int]:
        return self.seqs

    def _init_sensored_data(self):
        self.seqs = []
        self.sensored_data_list = []

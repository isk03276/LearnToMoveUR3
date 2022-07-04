import datetime

import numpy as np
from ray.rllib.agents.trainer import Trainer


def make_folder_name() -> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time


def make_initial_hidden_state(lstm_cell_size: int) -> list:
    """
    Make initial hidden state for testing lstm-based policy network.
    Args:
        lstm_cell_size (int): lstm cell size

    Returns:
        list: hidden state
    """
    hidden_state = [np.zeros(lstm_cell_size), np.zeros(lstm_cell_size)]
    return hidden_state


def save_model(trainer: Trainer, path_to_save: str):
    """
    Save trained model.
    Args:
        trainer (Trainer): rllib trainer
    """
    trainer.save(path_to_save)


def load_model(trainer: Trainer, path_to_load: str):
    """
    Load trained model.
    Args:
        trainer (Trainer): rllib trainer
        path_to_load (str): path to load
    """
    trainer.restore(path_to_load)

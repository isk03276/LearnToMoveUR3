import yaml


def load_config(config_file:str)-> dict:
    """
    Load a config file.

    Args:
        config_file (str): config file path

    Returns:
        (dict): loaded config infoes
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config    
    
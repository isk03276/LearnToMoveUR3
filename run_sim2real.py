import argparse

from utils.rllib import (
    load_model,
    make_initial_hidden_state,
    get_logger_creator,
    make_logging_folder,
    CustomLogCallback,
)
from utils.config import load_config, save_config
from envs.wrappers import ImageObsWrapper

import ray
from ray import tune
from ray.rllib.agents import ppo


def make_env(
    env_id: str, use_image_observation: bool, use_arm_camera: bool, rendering: bool
):
    if env_id == "reach":
        from envs import ReachEnv as env_class
    else:
        raise NotImplementedError

    env = env_class(use_arm_camera=use_arm_camera, rendering=rendering)
    return ImageObsWrapper(env) if use_image_observation else env


def get_interface_class(env_id):
    if env_id == "reach":
        from sim2real.interfaces.reach_interface import (
            ReachInterface as interface_class,
        )
    else:
        raise NotImplementedError
    return interface_class


def test(trainer, interface):
    state = interface.do_sensing()
    print(state)


def run(args):
    # load rllib config
    ray.init()
    configs = load_config(args.config_file_path)
    configs_to_save = configs.copy()
    rllib_configs = configs["rllib"]
    rllib_configs["callbacks"] = CustomLogCallback
    rllib_configs["num_workers"] = 1

    # env setting
    env_id = args.env_id
    env_config = configs["env_config"]
    env_args = {
        "env_id": env_id,
        "use_image_observation": env_config["use_image_observation"],
        "use_arm_camera": env_config["use_arm_camera"],
        "rendering": False,
    }
    tune.register_env(
        env_id,
        lambda _: make_env(**env_args),
    )

    # ros interface
    interface_class = get_interface_class(env_id)
    interface = interface_class(
        "/arm_joint_state", "/gripper_joint_state", "/object_poses", None, None
    )

    # logging setting
    logdir = make_logging_folder(root_dir="checkpoints/", env_id=env_id, is_test=True)
    save_config(configs_to_save, logdir + "/config.yaml")
    logger_creator = get_logger_creator(logdir=logdir)

    # rllib trainer setting
    trainer = ppo.PPOTrainer(
        env=env_id, config=rllib_configs, logger_creator=logger_creator
    )

    if args.load_from is not None:
        load_model(trainer, args.load_from)

    test(trainer, interface)

    ray.shutdown()


def go(args):
    interface_class = get_interface_class(args.env_id)
    interface = interface_class(
        "/arm_joint_state", "/gripper_joint_state", "/object_poses", None, None
    )
    print(interface.do_sensing())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROS interface for sim2real")
    parser.add_argument(
        "--env-id", default="reach", type=str, help="game environment id: 'reach', ..."
    )
    parser.add_argument(
        "--config-file-path",
        default="configs/default_config.yaml",
        type=str,
        help="Rllib config file path",
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")

    args = parser.parse_args()
    run(args)

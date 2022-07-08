import argparse

from utils.rllib import (
    load_model,
    make_initial_hidden_state,
    get_logger_creator,
    make_logging_folder,
    save_model,
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


def train(trainer, target_success_mean, path_to_save, save_interval):
    status = "[Train] {:2d} reward {:6.2f} len {:6.2f} success mean {:6.2f}"

    iteration = 0
    while True:
        result = trainer.train()
        success_mean = result["custom_metrics"]["success_mean"]
        print(
            status.format(
                iteration,
                result["episode_reward_mean"],
                result["episode_len_mean"],
                success_mean,
            )
        )
        iteration += 1
        if iteration % save_interval == 0:
            save_model(trainer, path_to_save)
        if success_mean >= target_success_mean:
            break


def test(env, trainer, test_num):
    use_lstm = trainer.config.get("model").get("use_lstm")
    lstm_cell_size = trainer.config.get("model").get("lstm_cell_size")
    success_list = []
    for ep in range(test_num):
        done = False
        obs = env.reset()
        rews = []
        if use_lstm:
            hidden_state = make_initial_hidden_state(lstm_cell_size)

        status = "[Test] {:2d} reward {:6.2f} len {:6.2f}, success mean {:6.2f}"

        while not done:
            if use_lstm:
                action, hidden_state, _ = trainer.compute_action(obs, hidden_state)
            else:
                action = trainer.compute_action(obs)
            obs, rew, done, info = env.step(action)
            rews.append(rew)
        success_list.append(int(info["success"]))
        print(
            status.format(
                ep + 1,
                sum(rews) / len(rews),
                len(rews),
                sum(success_list) / len(success_list),
            )
        )


def run(args):
    # load rllib config
    ray.init()
    configs = load_config(args.config_file_path)
    configs_to_save = configs.copy()
    rllib_configs = configs["rllib"]
    rllib_configs["callbacks"] = CustomLogCallback

    # env setting
    env_id = args.env_id
    env_config = configs["env_config"]
    env_args = {
        "env_id": env_id,
        "use_image_observation": env_config["use_image_observation"],
        "use_arm_camera": env_config["use_arm_camera"],
        "rendering": False if args.test else args.render,
    }
    tune.register_env(
        env_id, lambda _: make_env(**env_args),
    )

    # logging setting
    logdir = make_logging_folder(
        root_dir="checkpoints/", env_id=env_id, is_test=args.test
    )
    save_config(configs_to_save, logdir + "/config.yaml")
    logger_creator = get_logger_creator(logdir=logdir)

    # rllib trainer setting
    trainer = ppo.PPOTrainer(
        env=env_id, config=rllib_configs, logger_creator=logger_creator
    )

    if args.load_from is not None:
        load_model(trainer, args.load_from)

    if not args.test:
        train(trainer, args.target_success_mean, logdir, args.save_interval)

    env_args["rendering"] = args.render
    test_env = make_env(**env_args)
    test(test_env, trainer, args.test_num)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game environments to learn")
    parser.add_argument(
        "--env-id", default="reach", type=str, help="game environment id: 'reach', ..."
    )
    parser.add_argument(
        "--config-file-path",
        default="configs/default_config.yaml",
        type=str,
        help="Rllib config file path",
    )
    parser.add_argument("--render", action="store_true", help="Turn on rendering")
    # model
    parser.add_argument(
        "--save-interval", type=int, default=20, help="Model save interval"
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    # train/test
    parser.add_argument(
        "--target-success-mean",
        type=float,
        default=0.99,
        help="Learning ends when the current success mean is higher than ther target success mean",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--test-num", type=int, default=10, help="Number of episodes to test the model"
    )

    args = parser.parse_args()
    run(args)

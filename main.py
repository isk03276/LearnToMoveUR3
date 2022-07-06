import argparse

from utils.rllib import (
    load_model,
    make_initial_hidden_state,
    get_logger_creator,
    make_logging_folder,
    save_model,
    CustomLogCallback,
)
from utils.config import load_config

import ray
from ray import tune
from ray.rllib.agents import ppo


def get_env_generator(env_id: str):
    if env_id == "reach":
        from envs.reach.reach_env import ReachEnv as env_generator
    else:
        raise NotImplementedError
    return env_generator


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
    for ep in range(test_num):
        done = False
        obs = env.reset()
        rews = []
        success_list = []
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
    ray.init()
    env_id = args.env_id
    env_generator = get_env_generator(env_id)
    rllib_configs = load_config(args.config_file_path)
    rllib_configs["callbacks"] = CustomLogCallback
    use_arm_camera = rllib_configs["env_config"]["use_arm_camera"]
    tune.register_env(
        env_id,
        lambda _: env_generator(rendering=args.render, use_arm_camera=use_arm_camera),
    )
    logdir = make_logging_folder(
        root_dir="checkpoints/", env_id=env_id, is_test=args.test
    )
    logger_creator = get_logger_creator(logdir=logdir)
    trainer = ppo.PPOTrainer(
        env=env_id, config=rllib_configs, logger_creator=logger_creator
    )

    if args.load_from is not None:
        load_model(trainer, args.load_from)

    if not args.test:  # train
        train(trainer, args.target_success_mean, logdir, args.save_interval)

    test_env = env_generator(rendering=True, use_arm_camera=use_arm_camera)
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

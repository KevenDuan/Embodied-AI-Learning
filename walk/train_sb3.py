"""使用 SB3 PPO 训练教学版 Go2 站立策略。"""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

if __package__:
    from . import go2_env
    from .go2_observation import OBS_FIELDS
else:
    import go2_env
    from go2_observation import OBS_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, help="保存模型和日志的新目录")
    args = parser.parse_args()
    if args.timesteps <= 0 or not 0 <= args.seed < 2**32:
        parser.error("timesteps 须为正整数，seed 须在 [0, 2**32) 内")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.run_dir or Path(__file__).resolve().parent / "runs" / f"stand-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env_options = dict(decimation=10, max_episode_steps=500, action_scale=0.1,
                       reset_noise_scale=0.03, reset_tilt_deg=5.0)
    ppo_options = dict(n_steps=512, batch_size=64, n_epochs=5, learning_rate=3e-4,
                       gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                       policy_kwargs={"net_arch": [32, 32]}, device="cpu")
    # 小型 MLP 与单个 MuJoCo 环境用 CPU。限制线程以减少短矩阵运算的开销。
    torch.set_num_threads(1)
    env = go2_env.Go2Env(**env_options)
    monitored_env = Monitor(env, filename=str(run_dir / "train"))
    logger = configure(str(run_dir), ["stdout", "csv"])
    try:
        model = PPO("MlpPolicy", monitored_env, seed=args.seed, **ppo_options)
        model.set_logger(logger)
        model.learn(total_timesteps=args.timesteps)
        # SB3 通常在下一轮采样后记录上一轮更新，这里补上最后一轮的日志。
        logger.record("time/total_timesteps", model.num_timesteps)
        logger.dump(step=model.num_timesteps)
        model.save(run_dir / "ppo_go2_stand")

        config = {
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "ppo": ppo_options,
            "env": env_options,
            "control": {"kp": go2_env.KP, "kd": go2_env.KD,
                        "physics_dt": env.model.opt.timestep, "control_dt": env.control_dt},
            "reward": {"weights": go2_env.REWARD_WEIGHTS,
                       "reference_dt": go2_env.REWARD_REFERENCE_DT,
                       "home_height_m": env.home_height,
                       "height_width_m": go2_env.HEIGHT_REWARD_WIDTH,
                       "linear_speed_scale_m_s": go2_env.LINEAR_SPEED_SCALE,
                       "angular_speed_scale_rad_s": go2_env.ANGULAR_SPEED_SCALE,
                       "min_base_height_m": env.min_base_height,
                       "min_upright": env.min_upright},
            "observation": {
                "normalization": None,
                "fields": [{"name": name, "start": span.start, "stop": span.stop}
                           for name, span in OBS_FIELDS],
                "joint_order": [env.model.joint(j).name for j in env.model.actuator_trnid[:, 0]],
            },
            "versions": {name: version(name) for name in
                         ("stable-baselines3", "gymnasium", "mujoco", "numpy", "torch")},
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        logger.close()
        monitored_env.close()

    print(f"训练结束，模型和日志保存在 {run_dir.resolve()}")


if __name__ == "__main__":
    main()

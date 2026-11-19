"""加载 PPO 检查点，在相同初始扰动下对比零动作 PD。"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO

if __package__:
    from .go2_env import Go2Env
else:
    from go2_env import Go2Env


def evaluate_episode(env, choose_action, seed):
    obs, info = env.reset(seed=seed)
    initial_tilt = np.rad2deg(np.arccos(np.clip(info["upright"], -1, 1)))
    total_reward = 0.0
    total_physics_steps = 0
    measured_steps = 0
    tilt_sum = joint_error_sum = torque_sum = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(choose_action(obs))
        total_reward += reward
        substeps = info["physics_steps"]
        total_physics_steps += substeps
        if info["upright"] is not None and substeps:
            # 姿态和关节偏差取控制步末；力矩是整个控制步的子步均值。
            tilt_sum += np.rad2deg(np.arccos(np.clip(info["upright"], -1, 1))) * substeps
            joint_error_sum += np.mean(np.abs(obs[12:24])) * substeps
            torque_sum += info["mean_abs_torque_nm"] * substeps
            measured_steps += substeps
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "initial_tilt_deg": float(initial_tilt),
        "duration_s": total_physics_steps * env.model.opt.timestep,
        "mean_tilt_deg": float(tilt_sum / measured_steps) if measured_steps else None,
        "mean_joint_error_rad": float(joint_error_sum / measured_steps) if measured_steps else None,
        "mean_abs_torque_nm": float(torque_sum / measured_steps) if measured_steps else None,
        "return": total_reward,
        "end_reason": info["termination_reason"] or "time_limit",
    }


def show_replay(env, choose_action, seed):
    from mujoco import viewer

    obs, _ = env.reset(seed=seed)
    with viewer.launch_passive(env.model, env.data) as window:
        while window.is_running():
            start = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(choose_action(obs))
            window.sync()
            if terminated or truncated:
                break
            time.sleep(max(0.0, env.control_dt - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="train_sb3.py 生成的实验目录")
    parser.add_argument("--seed", type=int, default=1000, help="首个评估种子")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--reset-tilt-deg", type=float, default=8.0)
    parser.add_argument("--reset-noise-scale", type=float, default=0.04)
    parser.add_argument("--viewer", action="store_true", help="评估后用窗口回放首个种子")
    parser.add_argument("--report", type=Path, help="将逐回合指标写入 JSON")
    args = parser.parse_args()
    if args.seed < 0 or args.episodes <= 0:
        parser.error("seed 须为非负整数，episodes 须为正整数")

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    env_options = dict(config["env"], reset_tilt_deg=args.reset_tilt_deg,
                       reset_noise_scale=args.reset_noise_scale)
    env = Go2Env(**env_options)
    model = PPO.load(args.run_dir / "ppo_go2_stand.zip", device="cpu")

    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    seeds = range(args.seed, args.seed + args.episodes)
    policies = {"PPO": predict, "零动作 PD": lambda obs: zero}
    try:
        results = {name: [evaluate_episode(env, policy, seed) for seed in seeds]
                   for name, policy in policies.items()}
        print(f"评估种子 {list(seeds)}，机身倾斜上限 {env.reset_tilt_deg:g}°，"
              f"关节扰动 ±{env.reset_noise_scale:g} rad")
        metrics = {"duration_s": "时长(s)", "mean_tilt_deg": "倾角(°)",
                   "mean_joint_error_rad": "关节偏差(rad)",
                   "mean_abs_torque_nm": "绝对力矩(Nm)", "return": "回报"}
        for name, episodes in results.items():
            time_limits = sum(ep["end_reason"] == "time_limit" for ep in episodes)
            averages = []
            for key, label in metrics.items():
                values = [ep[key] for ep in episodes if ep[key] is not None]
                value = f"{np.mean(values):.4f}" if values else "无有效数据"
                averages.append(f"{label} {value}")
            print(f"{name}  正常超时 {time_limits}/{len(episodes)}  |  " + "  |  ".join(averages))

        if args.report is not None:
            report = {"checkpoint": str(args.run_dir / "ppo_go2_stand.zip"),
                      "env": env_options, "episodes": results}
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        if args.viewer:
            show_replay(env, predict, args.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()

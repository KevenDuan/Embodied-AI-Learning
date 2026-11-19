"""在仓库根目录运行 python walk/03_check_env.py，检查 Go2 环境。"""

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from go2_env import Go2Env


def run_demo(env, name, action_fn):
    """在相同初始状态下比较不同动作。"""
    env.reset(seed=0)
    min_height = float("inf")
    total_reward = 0.0
    while True:
        action = action_fn()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs) and np.isfinite(obs).all()
        total_reward += reward
        if info["base_height"] is not None:
            min_height = min(min_height, info["base_height"])
        if terminated or truncated:
            reason = info["termination_reason"] if terminated else "time_limit"
            print(f"{name}：{info['episode_steps']} 步，仿真 {info['sim_time']:.3f} s，"
                  f"控制步末最低机身高度 {min_height:.3f} m，"
                  f"总奖励 {total_reward:.1f}，结束原因 {reason}")
            return obs, terminated, truncated


def check_standing_reward():
    env = Go2Env(reset_noise_scale=0)
    zero = np.zeros(12, dtype=np.float32)
    scores = []
    for name, height, tilt_deg in (("站立", env.home_height, 0),
                                   ("倾斜", env.home_height, 35),
                                   ("低矮倒地", env.min_base_height - 0.01, 0)):
        env.reset(seed=0)
        start = env.base_qpos_id
        env.data.qpos[start + 2] = height
        mujoco.mju_quatIntegrate(env.data.qpos[start + 3:start + 7],
                                np.array([1.0, 0.0, 0.0]), np.deg2rad(tilt_deg))
        _, reward, terminated, _, info = env.step(zero)
        raw = info["reward_raw"]
        np.testing.assert_allclose(reward, sum(info["reward_weighted"].values()))
        assert terminated == (name == "低矮倒地")
        scores.append(reward)
        weighted = {key: round(value, 4) for key, value in info["reward_weighted"].items()}
        print(f"{name}奖励 {reward:.4f}，直立原始值 {raw['upright']:.4f}，"
              f"高度原始值 {raw['height']:.4f}，加权分项 {weighted}")
    assert scores[0] > scores[1] > scores[2]
    env.close()


def main():
    env = Go2Env(max_episode_steps=150)
    check_env(env)
    print("SB3 check_env 通过")
    print(f"物理步长 {env.model.opt.timestep:.3f} s，控制周期 {env.control_dt:.3f} s")

    zero = np.zeros(12, dtype=np.float32)
    small = zero.copy()
    small[1] = 0.2  # 右前腿大腿关节的目标角度偏移 0.02 rad。
    initial_obs, _ = env.reset(seed=0)
    obs, terminated, truncated = run_demo(env, "零动作", lambda: zero)
    assert not terminated and truncated, "零动作应保持站立，直到时间上限"
    obs, _, _ = run_demo(env, "单关节小动作", lambda: small)
    np.testing.assert_array_equal(obs[36:48], small)

    env.action_space.seed(0)
    run_demo(env, "随机动作", env.action_space.sample)

    reset_obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(reset_obs, initial_obs)
    np.testing.assert_array_equal(reset_obs[36:48], zero)
    assert info["episode_steps"] == 0 and info["sim_time"] == 0
    assert not env.data.ctrl.any()
    other_seed_obs, _ = env.reset(seed=1)
    assert not np.array_equal(other_seed_obs[12:24], reset_obs[12:24])
    print("重置检查通过：种子可复现，动作历史、力矩和计数均已清零")

    # 人为设一个下落状态，验证控制周期中途跌倒会打断剩余物理步。
    env.reset(seed=0)
    env.data.qpos[env.base_qpos_id + 2] = env.min_base_height + 0.002
    base_joint_id = env.model.body_jntadr[env.base_id]
    base_dof_id = env.model.jnt_dofadr[base_joint_id]
    env.data.qvel[base_dof_id + 2] = -2.0
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, truncated, info = env.step(zero)
    assert terminated and not truncated
    assert info["termination_reason"] == "low_base_height"
    assert 0 < info["physics_steps"] < env.decimation
    print(f"跌倒检查通过：仅执行 {info['physics_steps']}/{env.decimation} 个物理步")
    env.close()
    check_standing_reward()


if __name__ == "__main__":
    main()

"""读取 Go2 的 48 维观测；本步骤不包含强化学习训练。

运行（在仓库根目录）：
    python walk/02_observation.py
    python walk/02_observation.py --headless --duration 5
    python walk/02_observation.py --action-scale 0.1

默认零动作保持站姿。开启 action-scale 后，右前腿大腿关节做正弦运动。
速度、重力方向和速度指令均使用机身坐标系（x 前、y 左、z 上）。
"""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

if __package__:
    from .go2_observation import OBS_FIELDS, get_observation
else:
    from go2_observation import OBS_FIELDS, get_observation


KP = 200.0
KD = 5.0
FREQUENCY_HZ = 0.5
JOINT_INDEX = 1  # 按执行器顺序：FR_thigh_joint

def print_observation(sim_time, obs):
    print(f"\n仿真时间 {sim_time:.3f} s，shape={obs.shape}，dtype={obs.dtype}")
    for name, indices in OBS_FIELDS:
        values = np.array2string(obs[indices], precision=3, suppress_small=True)
        print(f"  [{indices.start:2d}:{indices.stop:2d}] {name}：{values}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="不打开窗口、不限速")
    parser.add_argument("--duration", type=float, default=10.0, help="仿真时长（秒）")
    parser.add_argument("--action-scale", type=float, default=0.0,
                        help="正弦目标角度偏移幅度（rad），默认 0：保持站姿")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration 必须为正的有限数值")
    if not np.isfinite(args.action_scale) or args.action_scale < 0:
        parser.error("action-scale 必须为非负有限数值")

    xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if home_id < 0 or base_id < 0:
        raise ValueError("模型必须包含 home 关键帧和 base_link 机身")
    if model.nu != 12 or np.linalg.norm(model.opt.gravity) == 0:
        raise ValueError("此教学脚本要求 12 个执行器和非零重力")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    joint_ids = model.actuator_trnid[:, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    q_default = data.qpos[qpos_ids].copy()
    joint_limits = model.jnt_range[joint_ids]
    command = np.zeros(3)  # [期望 vx, 期望 vy, 期望偏航角速度]
    last_action = np.zeros(model.nu)
    print("关节顺序：", [model.joint(j).name for j in joint_ids])
    print("速度是仿真中的真实值；速度指令只是观测输入，此处没有跟踪指令的策略。")
    print_observation(data.time, get_observation(
        model, data, base_id, qpos_ids, dof_ids, q_default, command, last_action
    ))

    def run(viewer=None):
        next_print_time = 1.0
        while data.time < args.duration:
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()
            action = np.zeros(model.nu)
            if args.action_scale > 0:
                action[JOINT_INDEX] = np.sin(2 * np.pi * FREQUENCY_HZ * data.time)
            action = np.clip(action, -1, 1)
            target_raw = q_default + args.action_scale * action
            target = np.clip(target_raw, joint_limits[:, 0], joint_limits[:, 1])
            torque = KP * (target - data.qpos[qpos_ids]) - KD * data.qvel[dof_ids]
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0],
                                   model.actuator_ctrlrange[:, 1])
            # 本步骤沿用 01：每个物理步更新动作；后续再引入 decimation。
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            # 新状态中的“上一步动作”就是刚刚执行的动作，初始时为零。
            last_action[:] = action
            obs = get_observation(model, data, base_id, qpos_ids, dof_ids,
                                  q_default, command, last_action)
            if data.time >= next_print_time:
                print_observation(data.time, obs)
                next_print_time += 1.0
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - wall_start)))

    if args.headless:
        run()
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            run(viewer)


if __name__ == "__main__":
    main()

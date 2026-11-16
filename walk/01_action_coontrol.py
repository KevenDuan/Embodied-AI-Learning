from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

KP = 200.0  # 比例增益：角度偏差越大，纠正力矩越大
KD = 5.0    # 微分增益：抑制关节运动，减小振荡
ACTION_SCALE = 0.5 # 动作值乘以此系数，得到角度偏移（rad）
FREQUENCY_HZ = 0.5 # 正弦动作频率


def main():
    xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("模型中未找到 home 关键帧")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    joint_ids = model.actuator_trnid[:, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    # 固定的默认站姿：每一步的目标都从这里计算，不累加动作。
    q_default = data.qpos[qpos_ids].copy()
    joint_limits = model.jnt_range[joint_ids]

    n = 1 # 需要操作的关节

    print(f"动作第 {n} 维对应关节：", model.joint(joint_ids[n]).name)
    print(f"角度偏移幅度：{ACTION_SCALE} rad，频率：{FREQUENCY_HZ} Hz")

    # 打开 viewer，不断计算力矩、推进仿真、刷新画面。
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 12 维动作，只改变第 n 个关节，其余关节保持默认目标。
            action = np.zeros(model.nu)
            action[n] = np.sin(2 * np.pi * FREQUENCY_HZ * data.time)
            # action = np.clip(action, -1.0, 1.0)

            target_raw = q_default + ACTION_SCALE * action
            # 当前模型中的 12 个电机关节均有限位；角度限幅与力矩限幅分别处理。
            target = np.clip(target_raw, joint_limits[:, 0], joint_limits[:, 1])

            q = data.qpos[qpos_ids]   # 当前角度
            dq = data.qvel[dof_ids]   # 当前角速度
            torque = KP * (target - q) - KD * dq

            # 当前模型 motor 的 gear=1，ctrl 就是力矩；限制在电机允许范围内。
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0],
                                   model.actuator_ctrlrange[:, 1])

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)  # 简单限速，便于观察


if __name__ == "__main__":
    main()

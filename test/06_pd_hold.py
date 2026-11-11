"""用最简单的 PD 控制循环，让 12 个关节保持 home 初始角度。

运行：python test/06_pd_hold.py
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


KP = 200.0  # 比例增益：角度偏差越大，纠正力矩越大
KD = 5.0    # 微分增益：抑制关节运动，减小振荡


def main():
    # 加载模型，恢复 home 初始状态。
    xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    # 按电机顺序，找到对应关节在 qpos 和 qvel 中的位置。
    joint_ids = model.actuator_trnid[:, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    target = data.qpos[qpos_ids].copy()  # 固定目标角度，目标速度为零

    print("初始目标角度（rad）：", target)
    print("PD 控制已启动，关闭 viewer 窗口退出。")

    # 打开 viewer，不断计算力矩、推进仿真、刷新画面。
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
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

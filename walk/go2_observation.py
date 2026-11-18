"""将 Go2 的 MuJoCo 状态组装成 48 维观测。"""

import mujoco
import numpy as np


def get_observation(model, data, base_id, qpos_ids, dof_ids,
                    q_default, command, last_action):
    """调用前先 mj_forward，保证派生状态与当前 qpos/qvel 一致。"""
    # 返回顺序是 [角速度(3), 线速度(3)]，与下面观测拼接顺序相反。
    # XBODY 指普通 body 坐标系（原点在 base_link 原点），local=1 使用其局部轴。
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, base_id, velocity, 1
    )
    angular_velocity = velocity[:3]
    linear_velocity = velocity[3:]

    # xmat 是机身 → 世界的旋转矩阵，其转置才是世界 → 机身。
    # MuJoCo 的四元数顺序是 [w, x, y, z]；这里直接用派生矩阵避免手写转换。
    rotation = data.xmat[base_id].reshape(3, 3)
    gravity_world = model.opt.gravity / np.linalg.norm(model.opt.gravity)
    gravity_body = rotation.T @ gravity_world

    obs = np.concatenate((
        linear_velocity,
        angular_velocity,
        gravity_body,
        command,
        data.qpos[qpos_ids] - q_default,
        data.qvel[dof_ids],
        last_action,
    )).astype(np.float32)

    if obs.shape != (48,) or not np.isfinite(obs).all():
        raise ValueError("观测必须是 48 维有限数值，请检查状态和输入")
    return obs

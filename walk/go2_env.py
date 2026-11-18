"""README 第 4 步：将 Go2 仿真封装成最小 Gymnasium 环境。"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

if __package__:
    from .go2_observation import get_observation
else:
    from go2_observation import get_observation


SCENE_PATH = Path(__file__).resolve().parents[1] / "xmls" / "scene_go2.xml"
KP = 200.0
KD = 5.0
NUMERIC_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


class Go2Env(gym.Env):
    """输入 12 维关节动作，输出 48 维观测。每个回合从 home 开始。"""

    def __init__(self, *, decimation=10, max_episode_steps=500,
                 action_scale=0.1, reset_noise_scale=0.01):
        super().__init__()
        for name, value in (("decimation", decimation),
                            ("max_episode_steps", max_episode_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        for name, value in (("action_scale", action_scale),
                            ("reset_noise_scale", reset_noise_scale)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数值")

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.home_id = self.model.key("home").id
        self.base_id = self.model.body("base_link").id
        base_joint_id = self.model.body_jntadr[self.base_id]
        self.base_qpos_id = self.model.jnt_qposadr[base_joint_id]

        # 用执行器顺序取关节状态，与 data.ctrl 的顺序保持一致。
        joint_ids = self.model.actuator_trnid[:, 0]
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.dof_ids = self.model.jnt_dofadr[joint_ids]
        self.q_default = self.model.key_qpos[self.home_id, self.qpos_ids].copy()
        self.joint_limits = self.model.jnt_range[joint_ids]
        self.torque_limits = self.model.actuator_ctrlrange

        self.decimation = decimation
        self.control_dt = self.model.opt.timestep * decimation
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.reset_noise_scale = reset_noise_scale

        # 平地 z=0。home 时机身 z=0.27 m，零动作站立约为 0.27～0.28 m。
        home_height = self.model.key_qpos[self.home_id, self.base_qpos_id + 2]
        self.min_base_height = 0.5 * home_height
        self.min_upright = np.cos(np.deg2rad(60.0))
        self.base_rotation = np.empty(9)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        # 沿用 02 的原始单位，速度等分量暂不设人为上下界。
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(48,), dtype=np.float32
        )
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.episode_steps = 0
        self.needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)

        # 先恢复 home，再给电机关节角度加可复现的小扰动。设为 0 可精确恢复 home。
        if self.reset_noise_scale > 0:
            noise = self.np_random.uniform(
                -self.reset_noise_scale, self.reset_noise_scale, size=12
            )
            self.data.qpos[self.qpos_ids] = np.clip(
                self.q_default + noise, self.joint_limits[:, 0], self.joint_limits[:, 1]
            )
        self.data.ctrl.fill(0)
        self.last_action.fill(0)
        self.command.fill(0)
        self.episode_steps = 0
        mujoco.mj_forward(self.model, self.data)

        obs = self._observation()
        self.last_valid_obs = obs.copy()
        self.needs_reset = False
        return obs, self._info(physics_steps=0)

    def step(self, action):
        if self.needs_reset:
            raise gym.error.ResetNeeded("请先调用 reset() 开始新回合")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("action 必须包含 12 个有限数值")
        action = np.clip(action, -1.0, 1.0)

        # 每个控制步生成一次目标角度。下面每个物理步用最新关节状态重算力矩。
        target = np.clip(self.q_default + self.action_scale * action,
                         self.joint_limits[:, 0], self.joint_limits[:, 1])
        reason = self._termination_reason()
        physics_steps = 0
        for _ in range(self.decimation):
            if reason is not None:
                break
            q = self.data.qpos[self.qpos_ids]
            dq = self.data.qvel[self.dof_ids]
            torque = KP * (target - q) - KD * dq
            self.data.ctrl[:] = np.clip(torque, self.torque_limits[:, 0],
                                       self.torque_limits[:, 1])
            previous_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            physics_steps += 1
            reason = self._termination_reason()
            if self.data.time <= previous_time:
                reason = "numerical_error"  # MuJoCo 检测到异常时可能自动重置仿真。

        self.episode_steps += 1
        terminated = reason is not None
        truncated = not terminated and self.episode_steps >= self.max_episode_steps
        self.needs_reset = terminated or truncated

        if reason == "numerical_error":
            # 无效状态无法构成观测。返回最近一次有效观测，并标记终止原因。
            obs = self.last_valid_obs.copy()
        else:
            # 用积分后的 qpos/qvel 刷新派生数据，只在取观测前做一次。
            if physics_steps:
                self.last_action[:] = action
            mujoco.mj_forward(self.model, self.data)
            obs = self._observation()
            self.last_valid_obs = obs.copy()

        # 第 4 步先用每控制步的存活分跑通接口，详细站立奖励放到第 5 步。
        reward = 0.0 if terminated else 1.0
        return obs, reward, terminated, truncated, self._info(physics_steps, reason)

    def _observation(self):
        return get_observation(
            self.model, self.data, self.base_id, self.qpos_ids, self.dof_ids,
            self.q_default, self.command, self.last_action,
        )

    def _base_height_and_upright(self):
        # 直接用物理步刚更新的浮动基座状态。xpos/xmat 等派生数据稍后才刷新。
        base_pose = self.data.qpos[self.base_qpos_id:self.base_qpos_id + 7]
        mujoco.mju_quat2Mat(self.base_rotation, base_pose[3:7])
        # R[2, 2] 是机身上方向与世界上方向的点积，即倾斜角的余弦。
        return float(base_pose[2]), float(self.base_rotation[8])

    def _termination_reason(self):
        if (not np.isfinite(self.data.time)
                or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or any(self.data.warning[w].number for w in NUMERIC_WARNINGS)):
            return "numerical_error"
        height, upright = self._base_height_and_upright()
        if height < self.min_base_height:
            return "low_base_height"
        if upright < self.min_upright:
            return "excessive_tilt"
        return None

    def _info(self, physics_steps, reason=None):
        height, upright = (None, None) if reason == "numerical_error" else (
            self._base_height_and_upright()
        )
        return {
            "base_height": height,
            "upright": upright,
            "episode_steps": self.episode_steps,
            "physics_steps": physics_steps,  # 本次 step 实际执行的子步数
            "sim_time": float(self.data.time),
            "termination_reason": reason,
        }

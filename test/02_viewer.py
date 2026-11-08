"""
加载机器狗模型并打开 MuJoCo viewer。
"""

from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
    xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # 使用 home 关键帧中的初始姿态（如果存在）。
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)

    # 此模型使用力矩电机；查看模型时将控制力矩归零。
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    print(f"已加载：{xml_path}")
    print("空格：暂停/继续仿真；关闭 viewer 窗口：退出。")
    print("当前未添加站立控制器，机器狗会在重力作用下自然倒下。")
    mujoco.viewer.launch(model=model, data=data)


if __name__ == "__main__":
    main()

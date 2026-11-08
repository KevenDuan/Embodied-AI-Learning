"""以表格形式打印初始状态下每个 body 的世界坐标 xpos。"""

from pathlib import Path

import mujoco


def main() -> None:
    xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # 与 02_viewer.py 一致，优先使用 home 关键帧初始化。
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    data.ctrl[:] = 0

    # 根据初始位姿计算所有 body 的世界坐标，不推进仿真时间。
    mujoco.mj_forward(model, data)

    print(f"模型：{xml_path}")
    print(f"body 数量：{model.nbody}（含 world），{model.nbody - 1}（不含 world）")
    print(f"初始状态来源：{'home 关键帧' if home_id >= 0 else '模型默认状态'}")
    print("xpos 为 body 原点在世界坐标系中的位置，单位：米。\n")

    header = f"{'ID':>3}  {'Body':<20} {'Parent':<20} {'xpos_x':>12} {'xpos_y':>12} {'xpos_z':>12}"
    print(header)
    print("-" * len(header))
    for body_id in range(model.nbody):
        name = model.body(body_id).name or f"unnamed_{body_id}"
        parent_id = int(model.body_parentid[body_id])
        parent_name = (model.body(parent_id).name or f"unnamed_{parent_id}") if body_id else "-"
        x, y, z = data.xpos[body_id]
        print(f"{body_id:>3}  {name:<20} {parent_name:<20} {x:>12.6f} {y:>12.6f} {z:>12.6f}")


if __name__ == "__main__":
    main()

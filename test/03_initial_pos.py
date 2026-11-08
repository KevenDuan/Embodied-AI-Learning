"""以表格形式打印模型中每个 body 的局部 pos 坐标。"""

from pathlib import Path

import mujoco


def main() -> None:
    xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"模型：{xml_path}")
    print(f"body 数量：{model.nbody}（含 world），{model.nbody - 1}（不含 world）")
    print("pos 为模型定义的相对于父 body 的局部坐标，单位：米。\n")

    header = f"{'ID':>3}  {'Body':<20} {'Parent':<20} {'pos_x':>12} {'pos_y':>12} {'pos_z':>12}"
    print(header)
    print("-" * len(header))
    for body_id in range(model.nbody):
        name = model.body(body_id).name or f"unnamed_{body_id}"
        parent_id = int(model.body_parentid[body_id])
        parent_name = (model.body(parent_id).name or f"unnamed_{parent_id}") if body_id else "-"
        x, y, z = model.body_pos[body_id]
        print(f"{body_id:>3}  {name:<20} {parent_name:<20} {x:>12.6f} {y:>12.6f} {z:>12.6f}")

    print("\n注：这里读取 model.body_pos，不是 home 初始状态下的世界坐标 data.xpos。")


if __name__ == "__main__":
    main()

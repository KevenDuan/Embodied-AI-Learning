"""
使用 MuJoCo 加载 xmls 中的模型，打印机器狗基本信息。
"""

from collections import Counter
from pathlib import Path
import mujoco


def print_robot_info(xml_path: Path) -> None:
    """分别统计每个模型，避免将两个 XML 中的同一台机器狗重复相加。"""
    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    free_count = int((model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum())

    print(f"\n{'=' * 64}")
    print(f"刚体数量（不含 world）：{model.nbody - 1}")
    print(f"关节总数（含浮动基座，njnt）：{model.njnt}")
    print(f"普通关节数量：{model.njnt - free_count}")
    print(f"浮动基座关节数量：{free_count}")
    print(f"速度自由度（nv）：{model.nv}")
    print(f"位置变量维度（nq）：{model.nq}")
    print(f"电机/执行器数量（nu）：{model.nu}")
    print(f"传感器数量：{model.nsensor}")
    print(f"传感器数据总维度：{model.nsensordata}")
    print(f"几何体数量（含场景地面等）：{model.ngeom}")
    print(f"模型总质量（不含 world）：{model.body_mass[1:].sum():.6f} kg")

    print("\n关节列表：")
    for index in range(model.njnt):
        name = model.joint(index).name or f"未命名_{index}"
        kind = mujoco.mjtJoint(int(model.jnt_type[index])).name.removeprefix("mjJNT_").lower()
        limits = ""
        if model.jnt_limited[index]:
            lower, upper = model.jnt_range[index]
            unit = "rad" if kind == "hinge" else "m"
            limits = f"，范围 [{lower:.5f}, {upper:.5f}] {unit}"
        print(f"  [{index}] {name} ({kind}){limits}")

    print("\n电机/执行器 → 关节：")
    for index in range(model.nu):
        name = model.actuator(index).name or f"未命名_{index}"
        transmission = mujoco.mjtTrn(int(model.actuator_trntype[index]))
        if transmission in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            joint_id = int(model.actuator_trnid[index, 0])
            target = model.joint(joint_id).name or f"未命名关节_{joint_id}"
        else:
            target = f"非关节传动 ({transmission.name})"
        print(f"  [{index}] {name} → {target}")
    if model.nu == 0:
        print("  此文件未定义执行器，不能据此推断实物没有电机。")

    if model.nsensor:
        print("\n传感器类型：")
        kinds = Counter(mujoco.mjtSensor(int(kind)).name for kind in model.sensor_type)
        for kind, count in kinds.items():
            print(f"  {kind}: {count}")


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "xmls" / "scene_go2.xml"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    try:
        print_robot_info(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"加载 {path} 时出错：{error}") from error


if __name__ == "__main__":
    main()

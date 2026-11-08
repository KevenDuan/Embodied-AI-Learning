"""按 04 的初始 xpos 绘制 body 三维骨架图。

运行：python 05_plot_body.py
仅保存图片：python 05_plot_body.py --no-show
点表示 body 原点，线表示父子关系，不表示模型的实际外形或碰撞几何。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="仅保存图片，不打开窗口")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    model = mujoco.MjModel.from_xml_path(str(directory.parent / "xmls" / "scene_go2.xml"))
    data = mujoco.MjData(model)

    # 与 04 一致：home 初始化，再计算世界坐标，不推进仿真。
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"FL": "#0072B2", "FR": "#D55E00", "RL": "#009E73", "RR": "#CC79A7"}
    labels = {"FL": "Front left", "FR": "Front right", "RL": "Rear left", "RR": "Rear right"}
    used_labels = set()

    for body_id in range(1, model.nbody):
        name = model.body(body_id).name or f"body_{body_id}"
        position = data.xpos[body_id]
        leg = name.split("_")[0]
        color = colors.get(leg, "#333333")
        label = labels.get(leg, "Base")
        ax.scatter(*position, color=color, s=45, depthshade=False,
                   label=label if label not in used_labels else None)
        used_labels.add(label)
        ax.text(*(position + np.array([0, 0, 0.009])), f"{body_id}: {name}", fontsize=8)

        parent_id = int(model.body_parentid[body_id])
        # world 是参考系，不画从世界原点到机身的连线。
        if parent_id != 0:
            segment = data.xpos[[parent_id, body_id]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=2)

    ax.scatter(0, 0, 0, marker="x", s=60, color="black", label="World origin")
    ax.text(0, 0, -0.025, "world", fontsize=8)
    # 地面参考网格 z=0。
    points = data.xpos
    lower = points.min(axis=0) - 0.06
    upper = points.max(axis=0) + 0.06
    grid_x, grid_y = np.meshgrid(np.linspace(lower[0], upper[0], 7),
                                 np.linspace(lower[1], upper[1], 7))
    ax.plot_wireframe(grid_x, grid_y, np.zeros_like(grid_x), color="gray", alpha=0.2, linewidth=0.6)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)  # 三个方向保持相同的米制比例。
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_zlabel("World Z (m)")
    ax.view_init(elev=24, azim=40)
    state = "home" if home_id >= 0 else "default"
    ax.set_title(f"Go2 body origins and parent-child links ({state} pose)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    output = directory / "05_body_xpos.png"
    fig.savefig(output, dpi=180)
    print(f"已保存三维 body 图：{output}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

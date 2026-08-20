#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 waypoint_tools CSV 航迹和关键任务点叠加到 S-FAST_LIO PCD 地图。"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MAPS = [
    Path("/home/user/fastlio_ws/src/S-FAST_LIO/PCD/GlobalMap_ikdtree.pcd"),
    Path("/home/user/fastlio_ws/src/S-FAST_LIO/PCD/GlobalMap.pcd"),
]


def read_pcd_xy(path):
    """读取 binary/ascii PCD 的 x/y；不依赖 open3d。"""
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PCD 缺少 DATA 行")
            header.append(line.decode("ascii", "replace").strip())
            if header[-1].lower().startswith("data "):
                break
        offset = handle.tell()
    values = {line.split(maxsplit=1)[0].upper(): line.split(maxsplit=1)[1]
              for line in header if len(line.split(maxsplit=1)) == 2}
    fields = values["FIELDS"].split()
    sizes = [int(v) for v in values["SIZE"].split()]
    types = values["TYPE"].split()
    count = int(values["POINTS"])
    data_kind = values["DATA"].lower()
    if "x" not in fields or "y" not in fields:
        raise ValueError("PCD 不含 x/y 字段")
    if data_kind == "binary":
        type_map = {"F": "f", "I": "i", "U": "u"}
        dtype = np.dtype([(field, "<" + type_map[kind] + str(size))
                          for field, size, kind in zip(fields, sizes, types)])
        points = np.fromfile(path, dtype=dtype, count=count, offset=offset)
        return points["x"], points["y"]
    if data_kind == "ascii":
        data = np.loadtxt(path, skiprows=len(header), usecols=(fields.index("x"), fields.index("y")))
        return data[:, 0], data[:, 1]
    raise ValueError("不支持的 PCD DATA 类型：%s" % data_kind)


def read_route(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or not {"x", "y"}.issubset(reader.fieldnames or set()):
        raise ValueError("CSV 为空或缺少 x/y 列")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_csv")
    parser.add_argument("--map", dest="map_file")
    parser.add_argument("--save", default="/tmp/route_map_check.png")
    args = parser.parse_args()
    route_path = Path(args.route_csv).expanduser()
    map_path = Path(args.map_file).expanduser() if args.map_file else next((p for p in DEFAULT_MAPS if p.exists()), None)
    if map_path is None or not map_path.exists():
        raise SystemExit("找不到 GlobalMap_ikdtree.pcd；请用 --map 指定地图。")
    rows = read_route(route_path)
    x, y = read_pcd_xy(map_path)
    stride = max(1, len(x) // 150000)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(x[::stride], y[::stride], s=0.15, c="#9ea7ad", alpha=0.45, rasterized=True, label="S-FAST_LIO 地图")
    rx = np.array([float(row["x"]) for row in rows])
    ry = np.array([float(row["y"]) for row in rows])
    ax.plot(rx, ry, "-", color="#0878c9", linewidth=1.3, label="录制航迹")
    groups = {"traffic_light": ("红绿灯", "#e53935", "D"), "piper_stop": ("机械臂任务点", "#7b1fa2", "o"), "avoid": ("避障区标记", "#fb8c00", "s"), "finish": ("终点", "#111111", "*")}
    used = set()
    for row in rows:
        task = (row.get("task") or "").strip()
        normalized = task[4:] if task.startswith("ext:") else task
        key = "traffic_light" if normalized == "traffic_light" else "piper_stop" if normalized.startswith("piper_stop_") else "avoid" if normalized in ("avoid_start", "avoid_end") else "finish" if normalized == "finish" else None
        if not key:
            continue
        label, color, marker = groups[key]
        px, py = float(row["x"]), float(row["y"])
        ax.scatter([px], [py], s=70, marker=marker, c=color, edgecolors="white", linewidths=0.8, label=label if key not in used else None, zorder=4)
        ax.annotate(normalized, (px, py), xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)
        used.add(key)
    ax.set_title("地图 / 录制航迹 / 关键点")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.save, dpi=180)
    print("已保存：%s" % args.save)
    plt.show()


if __name__ == "__main__":
    main()

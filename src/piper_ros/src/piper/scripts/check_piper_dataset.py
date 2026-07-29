#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import os
import sys

root = Path("/home/user/piper_ws/src/piper_ros/src/piper/scripts/dataset")

print("========== BASIC ==========")
print("dataset root:", root)
print("exists:", root.exists())
print("abs:", root.resolve() if root.exists() else "NOT FOUND")

if not root.exists():
    sys.exit(1)

print("\n========== FILE TREE ==========")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(p.relative_to(root), f"{p.stat().st_size / 1024:.2f} KB")

print("\n========== META CHECK ==========")
meta = root / "meta"
required_meta = [
    "info.json",
    "modality.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "stats.json",
]
for name in required_meta:
    p = meta / name
    print(name, "OK" if p.exists() else "MISSING")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if (meta / "info.json").exists():
    info = load_json(meta / "info.json")
    print("\n========== info.json SUMMARY ==========")
    print("codebase_version:", info.get("codebase_version"))
    print("robot_type:", info.get("robot_type"))
    print("fps:", info.get("fps"))
    print("total_episodes:", info.get("total_episodes"))
    print("total_frames:", info.get("total_frames"))
    print("total_videos:", info.get("total_videos"))
    print("data_path:", info.get("data_path"))
    print("video_path:", info.get("video_path"))
    print("features keys:")
    for k in info.get("features", {}).keys():
        print("  -", k)

    features = info.get("features", {})
    print("\n========== FEATURE DETAILS ==========")
    for key in ["observation.images.ego_view", "observation.state", "action"]:
        print("\n", key)
        print(json.dumps(features.get(key, None), indent=2, ensure_ascii=False))

if (meta / "modality.json").exists():
    modality = load_json(meta / "modality.json")
    print("\n========== modality.json ==========")
    print(json.dumps(modality, indent=2, ensure_ascii=False))

if (meta / "tasks.jsonl").exists():
    print("\n========== tasks.jsonl ==========")
    for line in (meta / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        print(line)

if (meta / "episodes.jsonl").exists():
    print("\n========== episodes.jsonl ==========")
    for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        print(line)

print("\n========== PARQUET CHECK ==========")
pq_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
print("num parquet:", len(pq_files))
for p in pq_files:
    print(p.relative_to(root), f"{p.stat().st_size / 1024:.2f} KB")

if pq_files:
    p = pq_files[0]
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(p)
        print("\nparquet file:", p.relative_to(root))
        print("num rows:", table.num_rows)
        print("columns:", table.column_names)
        print("schema:")
        print(table.schema)

        d = table.slice(0, min(3, table.num_rows)).to_pydict()
        print("\nfirst rows summary:")
        for k, v in d.items():
            first = v[0]
            if isinstance(first, list):
                print(k, "list_len=", len(first), "first_values=", first[:10])
            else:
                print(k, "first=", first)

        # 检查 action/state 维度
        full = table.to_pydict()
        if "observation.state" in full:
            state0 = full["observation.state"][0]
            print("\nstate dim:", len(state0))
        if "action" in full:
            action0 = full["action"][0]
            print("action dim:", len(action0))

        # 检查 done 最后一帧
        if "next.done" in full:
            done = full["next.done"]
            print("done first/mid/last:", done[0], done[len(done)//2], done[-1])
            print("done true count:", sum(bool(x) for x in done))

        # 检查 timestamp 是否递增
        if "timestamp" in full:
            ts = full["timestamp"]
            is_inc = all(ts[i] <= ts[i+1] for i in range(len(ts)-1))
            print("timestamp increasing:", is_inc)
            print("timestamp first/last:", ts[0], ts[-1])
            if len(ts) > 1:
                fps_est = (len(ts)-1) / max(ts[-1] - ts[0], 1e-6)
                print("estimated fps:", fps_est)

    except Exception as e:
        print("Failed to read parquet:", repr(e))

print("\n========== VIDEO CHECK ==========")
video_files = sorted((root / "videos").glob("chunk-*/*/episode_*.mp4"))
print("num videos:", len(video_files))
for p in video_files:
    print(p.relative_to(root), f"{p.stat().st_size / 1024:.2f} KB")

if video_files:
    try:
        import cv2
        vp = str(video_files[0])
        cap = cv2.VideoCapture(vp)
        ok = cap.isOpened()
        print("\nvideo open:", ok)
        if ok:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("video frames:", frame_count)
            print("video fps:", fps)
            print("video size:", width, height)
        cap.release()
    except Exception as e:
        print("Failed to check video:", repr(e))

print("\n========== EXPECTED ==========")
print("1. action/state dim should be 7: joint1-6 + gripper")
print("2. video should be RGB mp4, normally 256x256")
print("3. prompt should be: Pick up the red block and place it on the blue block.")
print("4. next.done should be True only on the last frame")
print("5. parquet rows should equal video frame count or be very close")

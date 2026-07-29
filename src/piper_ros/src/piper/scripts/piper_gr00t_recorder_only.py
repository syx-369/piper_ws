#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
piper_gr00t_recorder_only_fixed.py

ROS Noetic recorder for AgileX Piper teleoperation data, exported in a
GR00T-flavored LeRobot v2-like layout:

  dataset_root/
    meta/{info.json,tasks.jsonl,episodes.jsonl,modality.json}
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.ego_view/episode_000000.mp4

Recommended for master-slave teleoperation:
  action_source:=command
  action_topic:=/joint_ctrl_single        # or your teleop command topic to the slave arm

Fallback when command topic is not available:
  action_source:=future_state
  future_steps:=1                         # action[t] = state[t + 1]

Example:
python3 ~/piper_ws/src/piper_ros/src/piper/scripts/piper_gr00t_recorder_only_fixed.py \
  _camera_topic:=/camera/color/image_raw \
  _state_topic:=/joint_states_single \
  _action_source:=command \
  _action_topic:=/joint_ctrl_single \
  _dataset_root:=/home/user/datasets/piper_gr00t \
  '_task:=Pick up the red block and place it on the blue block.' \
  _fps:=30 \
  _resize_height:=256 \
  _resize_width:=256 \
  _max_age_ms:=50.0 \
  _action_max_age_ms:=150.0 \
  _keep_aspect_crop:=true

Press Ctrl-C to finish the current episode and write files.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    pa = None
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None


JOINT_NAMES = [
    "joint1_rad",
    "joint2_rad",
    "joint3_rad",
    "joint4_rad",
    "joint5_rad",
    "joint6_rad",
    "gripper_m",
]


@dataclass
class ImageSample:
    stamp: float
    frame_bgr: np.ndarray


@dataclass
class JointSample:
    stamp: float
    vec: np.ndarray


@dataclass
class PendingSample:
    stamp: float
    frame_bgr: np.ndarray
    state: np.ndarray
    action: Optional[np.ndarray] = None


class PiperGR00TRecorder:
    def __init__(self) -> None:
        rospy.init_node("piper_gr00t_recorder_only", anonymous=True)

        self.camera_topic: str = rospy.get_param("~camera_topic", "/camera/color/image_raw")
        self.state_topic: str = rospy.get_param("~state_topic", "/joint_states_single")
        self.action_topic: str = rospy.get_param("~action_topic", "/joint_ctrl_single")
        self.action_source: str = rospy.get_param("~action_source", "command")
        self.dataset_root = Path(rospy.get_param("~dataset_root", "/home/user/datasets/piper_gr00t")).expanduser()
        self.task: str = rospy.get_param("~task", "Pick up the object and place it at the target.")

        self.fps: float = float(rospy.get_param("~fps", 30.0))
        self.resize_h: int = int(rospy.get_param("~resize_height", 256))
        self.resize_w: int = int(rospy.get_param("~resize_width", 256))
        self.max_age_ms: float = float(rospy.get_param("~max_age_ms", 50.0))
        self.action_max_age_ms: float = float(rospy.get_param("~action_max_age_ms", 150.0))
        self.future_steps: int = int(rospy.get_param("~future_steps", 1))
        self.keep_aspect_crop: bool = bool(rospy.get_param("~keep_aspect_crop", True))
        self.clip_gripper: bool = bool(rospy.get_param("~clip_gripper", True))
        self.gripper_min: float = float(rospy.get_param("~gripper_min", 0.0))
        self.gripper_max: float = float(rospy.get_param("~gripper_max", 0.08))
        self.episode_seconds: float = float(rospy.get_param("~episode_seconds", 0.0))
        self.min_frames: int = int(rospy.get_param("~min_frames", 10))
        self.video_name: str = rospy.get_param("~video_name", "ego_view")
        self.robot_type: str = rospy.get_param("~robot_type", "agilex_piper_single")
        self.start_delay_sec: float = float(rospy.get_param("~start_delay_sec", 0.0))

        if self.action_source not in {"command", "future_state", "current_state"}:
            raise ValueError("~action_source must be one of: command, future_state, current_state")
        if self.fps <= 0:
            raise ValueError("~fps must be positive")
        if self.resize_h <= 0 or self.resize_w <= 0:
            raise ValueError("~resize_height and ~resize_width must be positive")
        if self.future_steps < 0:
            raise ValueError("~future_steps must be >= 0")
        if pa is None or pq is None:
            raise RuntimeError(
                "pyarrow is required to write parquet. Install it with: pip install pyarrow. "
                f"Original error: {_PYARROW_IMPORT_ERROR}"
            )

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.images: Deque[ImageSample] = deque(maxlen=300)
        self.states: Deque[JointSample] = deque(maxlen=2000)
        self.actions: Deque[JointSample] = deque(maxlen=2000)
        self.pending_future: Deque[PendingSample] = deque()

        self.rows: List[Dict] = []
        self.writer: Optional[cv2.VideoWriter] = None
        self.first_stamp: Optional[float] = None
        self.last_saved_image_stamp: Optional[float] = None
        self.dropped_no_image = 0
        self.dropped_no_state = 0
        self.dropped_no_action = 0
        self.dropped_duplicate_image = 0
        self.sample_attempts = 0

        self.episode_index = self._next_episode_index()
        self.global_index_start = self._existing_total_frames()
        self.task_index, self.validity_index = self._ensure_tasks()
        self.video_key = f"observation.images.{self.video_name}"
        self.video_path = (
            self.dataset_root
            / "videos"
            / "chunk-000"
            / self.video_key
            / f"episode_{self.episode_index:06d}.mp4"
        )
        self.data_path = (
            self.dataset_root
            / "data"
            / "chunk-000"
            / f"episode_{self.episode_index:06d}.parquet"
        )
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "meta").mkdir(parents=True, exist_ok=True)

        rospy.loginfo("GR00T recorder episode_index=%d", self.episode_index)
        rospy.loginfo("camera_topic=%s", self.camera_topic)
        rospy.loginfo("state_topic=%s", self.state_topic)
        rospy.loginfo("action_source=%s", self.action_source)
        if self.action_source == "command":
            rospy.loginfo("action_topic=%s", self.action_topic)
        rospy.loginfo("dataset_root=%s", str(self.dataset_root))
        rospy.loginfo("task=%s", self.task)

        rospy.Subscriber(self.camera_topic, Image, self._image_cb, queue_size=2, tcp_nodelay=True)
        rospy.Subscriber(self.state_topic, JointState, self._state_cb, queue_size=50, tcp_nodelay=True)
        if self.action_source == "command":
            rospy.Subscriber(self.action_topic, JointState, self._action_cb, queue_size=50, tcp_nodelay=True)

    # ------------------------- ROS callbacks -------------------------

    def _msg_stamp(self, msg) -> float:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None:
            t = stamp.to_sec()
            if t > 0:
                return float(t)
        return float(rospy.Time.now().to_sec())

    def _image_cb(self, msg: Image) -> None:
        try:
            # OpenCV VideoWriter expects BGR frames. cv_bridge converts most encodings safely.
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = self._resize_image(frame)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to convert image: %s", exc)
            return
        with self.lock:
            self.images.append(ImageSample(stamp=self._msg_stamp(msg), frame_bgr=frame))

    def _state_cb(self, msg: JointState) -> None:
        vec = self._joint_msg_to_vec(msg, label="state")
        if vec is None:
            return
        with self.lock:
            self.states.append(JointSample(stamp=self._msg_stamp(msg), vec=vec))

    def _action_cb(self, msg: JointState) -> None:
        vec = self._joint_msg_to_vec(msg, label="action")
        if vec is None:
            return
        with self.lock:
            self.actions.append(JointSample(stamp=self._msg_stamp(msg), vec=vec))

    # ------------------------- conversion helpers -------------------------

    def _joint_msg_to_vec(self, msg: JointState, label: str) -> Optional[np.ndarray]:
        if len(msg.position) < 7:
            rospy.logwarn_throttle(2.0, "%s JointState has len(position)=%d, need 7", label, len(msg.position))
            return None
        vec = np.asarray(list(msg.position[:7]), dtype=np.float32)
        if not np.all(np.isfinite(vec)):
            rospy.logwarn_throttle(2.0, "%s JointState contains NaN/Inf, skip", label)
            return None
        if self.clip_gripper:
            vec[6] = np.float32(np.clip(vec[6], self.gripper_min, self.gripper_max))
        return vec

    def _resize_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.keep_aspect_crop:
            return cv2.resize(frame_bgr, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        in_h, in_w = frame_bgr.shape[:2]
        scale = max(self.resize_w / float(in_w), self.resize_h / float(in_h))
        new_w = max(self.resize_w, int(round(in_w * scale)))
        new_h = max(self.resize_h, int(round(in_h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)
        x0 = max(0, (new_w - self.resize_w) // 2)
        y0 = max(0, (new_h - self.resize_h) // 2)
        crop = resized[y0 : y0 + self.resize_h, x0 : x0 + self.resize_w]
        if crop.shape[0] != self.resize_h or crop.shape[1] != self.resize_w:
            crop = cv2.resize(crop, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
        return crop

    def _nearest(self, samples: Iterable[JointSample], stamp: float, max_age_ms: float) -> Optional[JointSample]:
        max_age = max_age_ms / 1000.0
        best: Optional[JointSample] = None
        best_dt = float("inf")
        for sample in samples:
            dt = abs(sample.stamp - stamp)
            if dt < best_dt:
                best = sample
                best_dt = dt
        if best is None or best_dt > max_age:
            return None
        return best

    def _latest_before_or_near(self, samples: Iterable[JointSample], stamp: float, max_age_ms: float) -> Optional[JointSample]:
        """For command actions, prefer the latest command at or before image time.

        If all available commands are slightly after the image stamp, fall back to nearest.
        """
        max_age = max_age_ms / 1000.0
        latest_before: Optional[JointSample] = None
        latest_stamp = -float("inf")
        all_samples = list(samples)
        for sample in all_samples:
            if sample.stamp <= stamp and sample.stamp > latest_stamp:
                latest_before = sample
                latest_stamp = sample.stamp
        if latest_before is not None and (stamp - latest_before.stamp) <= max_age:
            return latest_before
        return self._nearest(all_samples, stamp, max_age_ms)

    # ------------------------- recording -------------------------

    def run(self) -> None:
        if self.start_delay_sec > 0:
            rospy.loginfo("Start delay %.2f sec", self.start_delay_sec)
            rospy.sleep(self.start_delay_sec)

        rate = rospy.Rate(self.fps)
        start_wall = time.time()
        try:
            while not rospy.is_shutdown():
                if self.episode_seconds > 0 and (time.time() - start_wall) >= self.episode_seconds:
                    rospy.loginfo("episode_seconds reached: %.3f", self.episode_seconds)
                    break
                self._sample_once()
                rate.sleep()
        finally:
            self._finalize()

    def _sample_once(self) -> None:
        self.sample_attempts += 1
        with self.lock:
            image = self.images[-1] if self.images else None
            states = list(self.states)
            actions = list(self.actions)

        if image is None:
            self.dropped_no_image += 1
            return
        if self.last_saved_image_stamp is not None and abs(image.stamp - self.last_saved_image_stamp) < 1e-6:
            self.dropped_duplicate_image += 1
            return

        state_sample = self._nearest(states, image.stamp, self.max_age_ms)
        if state_sample is None:
            self.dropped_no_state += 1
            return

        if self.action_source == "command":
            action_sample = self._latest_before_or_near(actions, image.stamp, self.action_max_age_ms)
            if action_sample is None:
                self.dropped_no_action += 1
                return
            self._emit_sample(PendingSample(image.stamp, image.frame_bgr, state_sample.vec, action_sample.vec))
        elif self.action_source == "current_state":
            self._emit_sample(PendingSample(image.stamp, image.frame_bgr, state_sample.vec, state_sample.vec.copy()))
        else:  # future_state
            self.pending_future.append(PendingSample(image.stamp, image.frame_bgr, state_sample.vec))
            while len(self.pending_future) > self.future_steps:
                current = self.pending_future.popleft()
                future = self.pending_future[self.future_steps - 1] if self.future_steps > 0 else current
                current.action = future.state.copy()
                self._emit_sample(current)

        self.last_saved_image_stamp = image.stamp

    def _emit_sample(self, sample: PendingSample) -> None:
        if sample.action is None:
            return
        if self.first_stamp is None:
            self.first_stamp = sample.stamp
            self._open_video_writer()
        assert self.first_stamp is not None

        frame_index = len(self.rows)
        timestamp = float(sample.stamp - self.first_stamp)
        global_index = int(self.global_index_start + frame_index)
        self._write_video_frame(sample.frame_bgr)
        self.rows.append(
            {
                "observation.state": sample.state.astype(np.float32).tolist(),
                "action": sample.action.astype(np.float32).tolist(),
                "timestamp": np.float32(timestamp),
                "frame_index": int(frame_index),
                "episode_index": int(self.episode_index),
                "index": global_index,
                "task_index": int(self.task_index),
                "annotation.human.action.task_description": int(self.task_index),
                "annotation.human.validity": int(self.validity_index),
                "next.reward": np.float32(0.0),
                "next.done": False,
            }
        )
        if frame_index > 0 and frame_index % int(max(1, self.fps * 5)) == 0:
            rospy.loginfo("recorded %d frames", frame_index)

    def _open_video_writer(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (self.resize_w, self.resize_h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")

    def _write_video_frame(self, frame_bgr: np.ndarray) -> None:
        if self.writer is None:
            self._open_video_writer()
        assert self.writer is not None
        self.writer.write(frame_bgr)

    # ------------------------- dataset files -------------------------

    def _next_episode_index(self) -> int:
        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        max_idx = -1
        if episodes_path.exists():
            with episodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        max_idx = max(max_idx, int(obj.get("episode_index", -1)))
                    except Exception:
                        pass
        data_dir = self.dataset_root / "data" / "chunk-000"
        if data_dir.exists():
            for p in data_dir.glob("episode_*.parquet"):
                try:
                    max_idx = max(max_idx, int(p.stem.split("_")[-1]))
                except Exception:
                    pass
        return max_idx + 1

    def _existing_total_frames(self) -> int:
        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        total = 0
        if episodes_path.exists():
            with episodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        total += int(json.loads(line).get("length", 0))
                    except Exception:
                        pass
        return total

    def _read_tasks(self) -> List[Dict]:
        tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
        tasks: List[Dict] = []
        if tasks_path.exists():
            with tasks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
        return tasks

    def _write_tasks(self, tasks: List[Dict]) -> None:
        tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        with tasks_path.open("w", encoding="utf-8") as f:
            for obj in sorted(tasks, key=lambda x: int(x["task_index"])):
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _ensure_tasks(self) -> Tuple[int, int]:
        tasks = self._read_tasks()
        next_idx = 0 if not tasks else max(int(t["task_index"]) for t in tasks) + 1

        def find_task(text: str) -> Optional[int]:
            for t in tasks:
                if t.get("task") == text:
                    return int(t["task_index"])
            return None

        task_idx = find_task(self.task)
        if task_idx is None:
            task_idx = next_idx
            next_idx += 1
            tasks.append({"task_index": task_idx, "task": self.task})

        valid_idx = find_task("valid")
        if valid_idx is None:
            valid_idx = next_idx
            tasks.append({"task_index": valid_idx, "task": "valid"})

        self._write_tasks(tasks)
        return task_idx, valid_idx

    def _write_modality_json(self) -> None:
        modality = {
            "state": {
                "single_arm": {"start": 0, "end": 6},
                "gripper": {"start": 6, "end": 7},
            },
            "action": {
                "single_arm": {"start": 0, "end": 6, "absolute": True},
                "gripper": {"start": 6, "end": 7, "absolute": True},
            },
            "video": {
                self.video_name: {"original_key": self.video_key},
            },
            "annotation": {
                "human.action.task_description": {},
                "human.validity": {},
            },
        }
        path = self.dataset_root / "meta" / "modality.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(modality, f, indent=2, ensure_ascii=False)

    def _append_episode_jsonl(self) -> None:
        path = self.dataset_root / "meta" / "episodes.jsonl"
        obj = {
            "episode_index": int(self.episode_index),
            "tasks": [int(self.task_index)],
            "length": int(len(self.rows)),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _write_info_json(self) -> None:
        episodes = []
        path = self.dataset_root / "meta" / "episodes.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                episodes = [json.loads(line) for line in f if line.strip()]
        tasks = self._read_tasks()
        total_episodes = len(episodes)
        total_frames = sum(int(e.get("length", 0)) for e in episodes)
        total_chunks = int(math.ceil(max(total_episodes, 1) / 1000.0))

        video_info = {
            "video.height": int(self.resize_h),
            "video.width": int(self.resize_w),
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": float(self.fps),
            "video.channels": 3,
            "has_audio": False,
        }
        info = {
            "codebase_version": "v2.0",
            "robot_type": self.robot_type,
            "total_episodes": int(total_episodes),
            "total_frames": int(total_frames),
            "total_tasks": int(len(tasks)),
            "total_videos": int(total_episodes),
            "total_chunks": int(total_chunks),
            "chunks_size": 1000,
            "fps": float(self.fps),
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                self.video_key: {
                    "dtype": "video",
                    "shape": [int(self.resize_h), int(self.resize_w), 3],
                    "names": ["height", "width", "channels"],
                    "info": video_info,
                    "video_info": video_info,
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": JOINT_NAMES,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": JOINT_NAMES,
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "annotation.human.action.task_description": {"dtype": "int64", "shape": [1], "names": None},
                "annotation.human.validity": {"dtype": "int64", "shape": [1], "names": None},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
            },
        }
        with (self.dataset_root / "meta" / "info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def _write_parquet(self) -> None:
        if not self.rows:
            return
        self.rows[-1]["next.done"] = True

        schema = pa.schema(
            [
                pa.field("observation.state", pa.list_(pa.float32(), list_size=7)),
                pa.field("action", pa.list_(pa.float32(), list_size=7)),
                pa.field("timestamp", pa.float32()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
                pa.field("annotation.human.action.task_description", pa.int64()),
                pa.field("annotation.human.validity", pa.int64()),
                pa.field("next.reward", pa.float32()),
                pa.field("next.done", pa.bool_()),
            ]
        )
        cols = {name: [row[name] for row in self.rows] for name in schema.names}
        table = pa.Table.from_pydict(cols, schema=schema)
        pq.write_table(table, self.data_path)

    def _finalize(self) -> None:
        # For future_state, remaining pending samples do not have a future target; drop them.
        if self.action_source == "future_state" and len(self.pending_future) > 0:
            rospy.loginfo("Dropped %d tail frames without future_state action", len(self.pending_future))
            self.pending_future.clear()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if len(self.rows) < self.min_frames:
            rospy.logwarn(
                "Only %d frames recorded (< min_frames=%d). Episode files will not be finalized.",
                len(self.rows),
                self.min_frames,
            )
            # Remove empty/too-short video file if OpenCV created one.
            try:
                if self.video_path.exists():
                    self.video_path.unlink()
            except Exception:
                pass
            return

        self._write_parquet()
        self._append_episode_jsonl()
        self._write_modality_json()
        self._write_info_json()

        rospy.loginfo("Saved parquet: %s", str(self.data_path))
        rospy.loginfo("Saved video:   %s", str(self.video_path))
        rospy.loginfo("Frames: %d", len(self.rows))
        rospy.loginfo(
            "Drop stats: attempts=%d no_image=%d duplicate_image=%d no_state=%d no_action=%d",
            self.sample_attempts,
            self.dropped_no_image,
            self.dropped_duplicate_image,
            self.dropped_no_state,
            self.dropped_no_action,
        )


def main() -> None:
    recorder = PiperGR00TRecorder()
    recorder.run()


if __name__ == "__main__":
    main()

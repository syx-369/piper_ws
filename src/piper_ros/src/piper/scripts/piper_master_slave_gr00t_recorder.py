#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 Piper master-slave teleoperation + GR00T-N1.5/LeRobot-style recorder.

功能：
1. 订阅主臂 JointState，映射成从臂 action。
2. 发布从臂控制 JointState 到 /joint_states。
3. 订阅从臂真实 JointState 作为 observation.state。
4. 订阅相机 /camera/color/image_raw。
5. 使用图像时间戳对齐 action/state。
6. 通过 ROS service 控制 episode:
   /start_episode
   /save_episode
   /discard_episode
   /finalize_dataset

默认任务 prompt:
"Pick up the red block and place it on the blue block."

数据格式：
root/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.ego_view/episode_000000.mp4
  meta/info.json
  meta/modality.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/stats.json
"""

import json
import math
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger, TriggerResponse

try:
    import cv2
except Exception as exc:
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None
    try:
        import pandas as pd
    except Exception:
        pd = None
else:
    pd = None


# =========================
# Utility
# =========================

def stamp_to_ns(stamp) -> int:
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)


def now_ns() -> int:
    return int(rospy.Time.now().secs) * 1_000_000_000 + int(rospy.Time.now().nsecs)


def array_to_json(x):
    arr = np.asarray(x)
    if arr.size == 1:
        v = arr.reshape(-1)[0]
        if arr.dtype.kind in ("i", "u"):
            return int(v)
        if arr.dtype.kind == "b":
            return bool(v)
        return float(v)
    if arr.dtype.kind in ("i", "u"):
        return [int(v) for v in arr.tolist()]
    if arr.dtype.kind == "b":
        return [bool(v) for v in arr.tolist()]
    return [float(v) for v in arr.tolist()]


# =========================
# Low-dimensional buffer
# =========================

class JointStateBuffer:
    def __init__(self, maxlen: int = 300):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, msg: JointState):
        if len(msg.position) == 0:
            return
        item = {
            "stamp_ns": stamp_to_ns(msg.header.stamp) if msg.header.stamp else now_ns(),
            "name": list(msg.name),
            "position": np.asarray(msg.position, dtype=np.float64).copy(),
        }
        with self.lock:
            self.buf.append(item)

    def ready(self) -> bool:
        with self.lock:
            return len(self.buf) > 0

    def latest(self):
        with self.lock:
            if not self.buf:
                return None
            return dict(self.buf[-1])

    def nearest(self, target_ns: int):
        with self.lock:
            if not self.buf:
                return None
            item = min(self.buf, key=lambda x: abs(x["stamp_ns"] - target_ns))
            return dict(item)

    def latest_before_or_nearest(self, target_ns: int):
        with self.lock:
            if not self.buf:
                return None
            prev = None
            for item in reversed(self.buf):
                if item["stamp_ns"] <= target_ns:
                    prev = item
                    break
            if prev is None:
                prev = min(self.buf, key=lambda x: abs(x["stamp_ns"] - target_ns))
            return dict(prev)

    def interpolate(self, target_ns: int):
        with self.lock:
            if not self.buf:
                return None
            if len(self.buf) == 1:
                return dict(self.buf[-1])

            prev = None
            next_ = None
            for item in self.buf:
                if item["stamp_ns"] <= target_ns:
                    prev = item
                if item["stamp_ns"] >= target_ns:
                    next_ = item
                    break

            if prev is None:
                return dict(self.buf[0])
            if next_ is None:
                return dict(self.buf[-1])
            if prev["stamp_ns"] == next_["stamp_ns"]:
                return dict(prev)
            if prev["name"] != next_["name"]:
                return self.nearest(target_ns)

            t0 = prev["stamp_ns"]
            t1 = next_["stamp_ns"]
            alpha = float(target_ns - t0) / float(t1 - t0)
            pos = (1.0 - alpha) * prev["position"] + alpha * next_["position"]
            return {
                "stamp_ns": int(target_ns),
                "name": list(prev["name"]),
                "position": pos.astype(np.float64),
            }


# =========================
# Statistics
# =========================

class RunningStats:
    def __init__(self):
        self.data = {}

    def update(self, name: str, values: np.ndarray):
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            raise ValueError(f"unsupported stats ndim={arr.ndim} for {name}")

        arr = arr.astype(np.float64)
        cur_min = np.min(arr, axis=0)
        cur_max = np.max(arr, axis=0)
        cur_sum = np.sum(arr, axis=0)
        cur_sumsq = np.sum(arr * arr, axis=0)
        cur_count = int(arr.shape[0])

        if name not in self.data:
            self.data[name] = {
                "min": cur_min,
                "max": cur_max,
                "sum": cur_sum,
                "sumsq": cur_sumsq,
                "count": cur_count,
            }
        else:
            s = self.data[name]
            s["min"] = np.minimum(s["min"], cur_min)
            s["max"] = np.maximum(s["max"], cur_max)
            s["sum"] = s["sum"] + cur_sum
            s["sumsq"] = s["sumsq"] + cur_sumsq
            s["count"] = int(s["count"]) + cur_count

    def to_jsonable(self):
        out = {}
        for name, s in self.data.items():
            count = max(int(s["count"]), 1)
            mean = s["sum"] / count
            var = np.maximum(s["sumsq"] / count - mean * mean, 0.0)
            std = np.sqrt(var)
            out[name] = {
                "mean": array_to_json(mean),
                "std": array_to_json(std),
                "min": array_to_json(s["min"]),
                "max": array_to_json(s["max"]),
            }
        return out


# =========================
# Video writer
# =========================

class StreamingVideoWriter:
    def __init__(self, path: Path, fps: int, resize_hw: Optional[tuple] = None):
        if cv2 is None:
            raise RuntimeError(f"cv2 import failed: {_cv2_import_error}")
        self.path = Path(path)
        self.fps = int(fps)
        self.resize_hw = resize_hw
        self.writer = None
        self.width = None
        self.height = None
        self.codec_name = "mp4v"

    def _open(self, width: int, height: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        candidates = [
            ("avc1", "h264"),
            ("mp4v", "mp4v"),
            ("H264", "h264"),
        ]
        for fourcc_name, codec_name in candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), (width, height))
            if writer is not None and writer.isOpened():
                self.writer = writer
                self.width = width
                self.height = height
                self.codec_name = codec_name
                return
            if writer is not None:
                writer.release()
        raise RuntimeError("failed to create video writer")

    def write_rgb(self, rgb: np.ndarray):
        if self.resize_hw is not None:
            h, w = self.resize_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected rgb HWC image, got {rgb.shape}")

        h, w, _ = rgb.shape
        if self.writer is None:
            self._open(w, h)

        if w != self.width or h != self.height:
            raise ValueError(f"image size changed: got {(h, w)}, expected {(self.height, self.width)}")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


@dataclass
class EpisodeBuffer:
    image_stamps_ns: list
    actions: list
    states: list


# =========================
# Main node
# =========================

class PiperMasterSlaveGr00tRecorder:
    def __init__(self):
        rospy.init_node("piper_master_slave_gr00t_recorder", anonymous=False)

        # Topics
        self.master_state_topic = rospy.get_param("~master_state_topic", "/master/joint_states_single")
        self.follower_state_topic = rospy.get_param("~follower_state_topic", "/joint_states_single")
        self.follower_cmd_topic = rospy.get_param("~follower_cmd_topic", "/joint_states")
        self.camera_topic = rospy.get_param("~camera_topic", "/camera/color/image_raw")

        # Dataset
        self.dataset_root = Path(rospy.get_param("~dataset_root", "/workspace/lmz/datasets/piper_red_block_on_blue_block"))
        self.task = rospy.get_param("~task", "Pick up the red block and place it on the blue block.")
        self.robot_type = rospy.get_param("~robot_type", "piper_single_arm")
        self.camera_key = rospy.get_param("~camera_key", "ego_view")
        self.fps = int(rospy.get_param("~fps", 20))
        self.resize_height = int(rospy.get_param("~resize_height", 256))
        self.resize_width = int(rospy.get_param("~resize_width", 256))
        self.chunk_size = int(rospy.get_param("~chunk_size", 1000))
        self.max_age_ms = float(rospy.get_param("~max_age_ms", 30.0))
        self.buffer_size = int(rospy.get_param("~buffer_size", 500))

        # Teleop
        self.control_rate_hz = float(rospy.get_param("~control_rate_hz", 50.0))
        self.command_speed = float(rospy.get_param("~command_speed", 15.0))
        self.enable_teleop = bool(rospy.get_param("~enable_teleop", True))
        self.mirror_joints = bool(rospy.get_param("~mirror_joints", False))
        self.gripper_scale = float(rospy.get_param("~gripper_scale", 1.0))
        self.max_joint_step_rad = float(rospy.get_param("~max_joint_step_rad", 0.04))
        self.max_gripper_step_m = float(rospy.get_param("~max_gripper_step_m", 0.004))

        # Optional offsets: follower_cmd = master * sign + offset
        self.joint_sign = np.asarray(rospy.get_param("~joint_sign", [1, 1, 1, 1, 1, 1, 1]), dtype=np.float64)
        self.joint_offset = np.asarray(rospy.get_param("~joint_offset", [0, 0, 0, 0, 0, 0, 0]), dtype=np.float64)
        if self.mirror_joints:
            self.joint_sign[:6] *= -1.0

        self.action_names = [f"joint_{i}" for i in range(6)] + ["gripper"]
        self.state_names = [f"joint_{i}" for i in range(6)] + ["gripper"]

        self.bridge = CvBridge()
        self.frame_period_ns = int(1e9 / self.fps)

        self.master_buf = JointStateBuffer(self.buffer_size)
        self.action_buf = JointStateBuffer(self.buffer_size)
        self.state_buf = JointStateBuffer(self.buffer_size)

        self.latest_image_msg = None
        self.latest_master = None
        self.latest_follower_state = None
        self.last_sent_action = None

        self.recording = False
        self.current_episode = None
        self.current_video_writer = None
        self.current_video_temp_path = None
        self.episode_start_ns = None
        self.last_recorded_image_ns = None
        self.frame_idx = 0
        self.global_index = self._discover_global_index()
        self.episode_idx = self._discover_next_episode_index()
        self.video_codec = "mp4v"

        self.global_stats = RunningStats()

        self.recorded_frames = 0
        self.dropped_not_ready = 0
        self.dropped_fps = 0
        self.dropped_sync = 0

        # ROS pub/sub
        self.cmd_pub = rospy.Publisher(self.follower_cmd_topic, JointState, queue_size=1)
        self.teleop_action_pub = rospy.Publisher("/teleop/arm_action", JointState, queue_size=10)
        self.robot_state_pub = rospy.Publisher("/robot/arm_state", JointState, queue_size=10)

        rospy.Subscriber(self.master_state_topic, JointState, self.master_state_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.follower_state_topic, JointState, self.follower_state_cb, queue_size=20, tcp_nodelay=True)
        rospy.Subscriber(self.camera_topic, Image, self.camera_cb, queue_size=5, tcp_nodelay=True)

        rospy.Service("/start_episode", Trigger, self.start_episode_cb)
        rospy.Service("/save_episode", Trigger, self.save_episode_cb)
        rospy.Service("/discard_episode", Trigger, self.discard_episode_cb)
        rospy.Service("/finalize_dataset", Trigger, self.finalize_dataset_cb)

        self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate_hz), self.control_timer_cb)
        self.log_timer = rospy.Timer(rospy.Duration(5.0), self.log_timer_cb)

        rospy.loginfo("Piper master-slave GR00T recorder started")
        rospy.loginfo(f"master_state_topic={self.master_state_topic}")
        rospy.loginfo(f"follower_state_topic={self.follower_state_topic}")
        rospy.loginfo(f"follower_cmd_topic={self.follower_cmd_topic}")
        rospy.loginfo(f"camera_topic={self.camera_topic}")
        rospy.loginfo(f"dataset_root={self.dataset_root}")
        rospy.loginfo(f"task={self.task}")
        rospy.loginfo(f"next_episode={self.episode_idx}")

    # ---------- ROS callbacks ----------

    def master_state_cb(self, msg: JointState):
        self.master_buf.push(msg)
        item = self.master_buf.latest()
        if item is not None:
            self.latest_master = item

    def follower_state_cb(self, msg: JointState):
        fixed = self._normalize_joint_msg(msg, source_name="follower_state")
        self.state_buf.push(fixed)
        self.robot_state_pub.publish(fixed)
        self.latest_follower_state = fixed

    def camera_cb(self, msg: Image):
        self.latest_image_msg = msg
        if self.recording:
            self.try_record_with_image(msg)

    def control_timer_cb(self, _event):
        if not self.enable_teleop:
            return
        if self.latest_master is None:
            return

        action = self._map_master_to_follower(self.latest_master["position"])
        if action is None:
            return

        action = self._smooth_action(action)
        self.last_sent_action = action.copy()

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(self.action_names)
        msg.position = [float(x) for x in action.tolist()]
        msg.velocity = [0.0] * 6 + [self.command_speed]
        msg.effort = [0.0] * 7

        self.cmd_pub.publish(msg)

        # 给 recorder 对齐用的 action topic。
        self.teleop_action_pub.publish(msg)
        self.action_buf.push(msg)

    def log_timer_cb(self, _event):
        if self.recording:
            rospy.loginfo(
                f"recording episode={self.episode_idx}, frames={self.recorded_frames}, "
                f"dropped_not_ready={self.dropped_not_ready}, "
                f"dropped_fps={self.dropped_fps}, dropped_sync={self.dropped_sync}"
            )

    # ---------- teleop mapping ----------

    def _normalize_joint_msg(self, msg: JointState, source_name: str) -> JointState:
        arr = np.asarray(msg.position, dtype=np.float64)
        if arr.size < 6:
            rospy.logwarn_throttle(2.0, f"{source_name}: expected at least 6 joints, got {arr.size}")
            return msg

        out = np.zeros(7, dtype=np.float64)
        out[:6] = arr[:6]
        out[6] = arr[6] if arr.size >= 7 else 0.0
        out[6] = float(np.clip(out[6], 0.0, 0.08))

        new_msg = JointState()
        new_msg.header.stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        new_msg.name = list(self.state_names)
        new_msg.position = [float(x) for x in out.tolist()]
        new_msg.velocity = list(msg.velocity[:7]) if len(msg.velocity) >= 7 else []
        new_msg.effort = list(msg.effort[:7]) if len(msg.effort) >= 7 else []
        return new_msg

    def _map_master_to_follower(self, master_position: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(master_position, dtype=np.float64)
        if arr.size < 6:
            return None

        out = np.zeros(7, dtype=np.float64)
        out[:6] = arr[:6]

        if arr.size >= 7:
            out[6] = arr[6] * self.gripper_scale
        else:
            out[6] = 0.0

        out = out * self.joint_sign + self.joint_offset
        out[6] = float(np.clip(out[6], 0.0, 0.08))
        return out

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if self.last_sent_action is None:
            return action

        prev = self.last_sent_action
        out = prev.copy()

        delta = action - prev
        delta[:6] = np.clip(delta[:6], -self.max_joint_step_rad, self.max_joint_step_rad)
        delta[6] = np.clip(delta[6], -self.max_gripper_step_m, self.max_gripper_step_m)

        out = prev + delta
        out[6] = float(np.clip(out[6], 0.0, 0.08))
        return out

    # ---------- recording ----------

    def ready(self) -> bool:
        return (
            self.latest_image_msg is not None
            and self.action_buf.ready()
            and self.state_buf.ready()
        )

    def get_synced_lowdim(self, image_ns: int):
        action_item = self.action_buf.latest_before_or_nearest(image_ns)
        state_item = self.state_buf.interpolate(image_ns)

        if action_item is None or state_item is None:
            return None

        action_skew_ms = abs(image_ns - action_item["stamp_ns"]) / 1_000_000.0
        state_skew_ms = abs(image_ns - state_item["stamp_ns"]) / 1_000_000.0

        if action_skew_ms > self.max_age_ms:
            rospy.logwarn_throttle(2.0, f"drop frame: action skew {action_skew_ms:.2f} ms > {self.max_age_ms}")
            return None

        if state_skew_ms > self.max_age_ms:
            rospy.logwarn_throttle(2.0, f"drop frame: state skew {state_skew_ms:.2f} ms > {self.max_age_ms}")
            return None

        action = np.asarray(action_item["position"], dtype=np.float64)
        state = np.asarray(state_item["position"], dtype=np.float64)

        if action.size != 7 or state.size != 7:
            rospy.logwarn_throttle(2.0, f"action/state dim mismatch: action={action.size}, state={state.size}")
            return None

        return {
            "action": action,
            "state": state,
        }

    def try_record_with_image(self, image_msg: Image):
        if not self.ready() or self.current_episode is None or self.current_video_writer is None:
            self.dropped_not_ready += 1
            return

        image_ns = stamp_to_ns(image_msg.header.stamp)

        if self.last_recorded_image_ns is not None:
            if image_ns - self.last_recorded_image_ns < self.frame_period_ns:
                self.dropped_fps += 1
                return

        synced = self.get_synced_lowdim(image_ns)
        if synced is None:
            self.dropped_sync += 1
            return

        rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        self.current_video_writer.write_rgb(rgb)
        self.video_codec = self.current_video_writer.codec_name

        if self.episode_start_ns is None:
            self.episode_start_ns = int(image_ns)

        self.current_episode.image_stamps_ns.append(int(image_ns))
        self.current_episode.actions.append(synced["action"].astype(np.float64).copy())
        self.current_episode.states.append(synced["state"].astype(np.float64).copy())

        self.last_recorded_image_ns = int(image_ns)
        self.frame_idx += 1
        self.recorded_frames += 1

    # ---------- services ----------

    def start_episode_cb(self, _req):
        res = TriggerResponse()
        try:
            if self.recording:
                raise RuntimeError("episode already recording")
            if not self.ready():
                raise RuntimeError(
                    "not ready: waiting for camera image, teleop action, and follower state"
                )

            self.ensure_dataset_layout()

            tmp_dir = self.dataset_root / ".tmp_videos"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = tempfile.NamedTemporaryFile(
                prefix=f"episode_{self.episode_idx:06d}_",
                suffix=".mp4",
                dir=str(tmp_dir),
                delete=False,
            )
            tmp_path = Path(tmp_file.name)
            tmp_file.close()

            self.current_video_temp_path = tmp_path
            self.current_video_writer = StreamingVideoWriter(
                tmp_path,
                fps=self.fps,
                resize_hw=(self.resize_height, self.resize_width),
            )
            self.current_episode = EpisodeBuffer([], [], [])

            self.frame_idx = 0
            self.recorded_frames = 0
            self.dropped_not_ready = 0
            self.dropped_fps = 0
            self.dropped_sync = 0
            self.episode_start_ns = None
            self.last_recorded_image_ns = None
            self.recording = True

            res.success = True
            res.message = f"episode {self.episode_idx} started"
            rospy.loginfo(res.message)
        except Exception as exc:
            res.success = False
            res.message = str(exc)
            rospy.logerr(f"start_episode failed: {exc}")
        return res

    def save_episode_cb(self, _req):
        res = TriggerResponse()
        try:
            self.recording = False

            if self.current_episode is None or self.frame_idx == 0:
                res.success = False
                res.message = "no frames recorded"
                return res

            self.current_video_writer.close()
            self.current_video_writer = None

            episode_index = int(self.episode_idx)
            task_index = 0

            data_path = self.data_path_for_episode(episode_index)
            video_path = self.video_path_for_episode(episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(self.current_video_temp_path), str(video_path))
            self.current_video_temp_path = None

            arrays = self.build_episode_arrays(episode_index, task_index)
            self.write_parquet(data_path, arrays)
            self.update_stats(arrays)
            self.append_episode_jsonl(episode_index, self.frame_idx)
            self.write_stats_json()
            self.write_info_json()
            self.write_modality_json()
            self.write_tasks_jsonl()

            res.success = True
            res.message = f"episode {episode_index} saved with {self.frame_idx} frames"
            rospy.loginfo(res.message)

            self.episode_idx += 1
            self.frame_idx = 0
            self.current_episode = None
            self.last_recorded_image_ns = None
            self.episode_start_ns = None
        except Exception as exc:
            res.success = False
            res.message = str(exc)
            rospy.logerr(f"save_episode failed: {exc}")
            self.cleanup_partial()
        return res

    def discard_episode_cb(self, _req):
        res = TriggerResponse()
        self.recording = False
        self.frame_idx = 0
        self.current_episode = None
        self.last_recorded_image_ns = None
        self.episode_start_ns = None

        if self.current_video_writer is not None:
            self.current_video_writer.close()
            self.current_video_writer = None

        if self.current_video_temp_path is not None and self.current_video_temp_path.exists():
            self.current_video_temp_path.unlink()
            self.current_video_temp_path = None

        res.success = True
        res.message = "episode discarded"
        rospy.loginfo(res.message)
        return res

    def finalize_dataset_cb(self, _req):
        res = TriggerResponse()
        try:
            self.ensure_dataset_layout()
            self.write_info_json()
            self.write_modality_json()
            self.write_tasks_jsonl()
            self.write_stats_json()
            res.success = True
            res.message = "dataset finalized"
        except Exception as exc:
            res.success = False
            res.message = str(exc)
        return res

    # ---------- dataset files ----------

    def ensure_dataset_layout(self):
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "meta").mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "data").mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "videos").mkdir(parents=True, exist_ok=True)

        self.write_tasks_jsonl()
        self.write_info_json()
        self.write_modality_json()
        self.write_stats_json()

        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            episodes_path.write_text("", encoding="utf-8")

    def _discover_next_episode_index(self) -> int:
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            return 0
        max_idx = -1
        for p in data_dir.glob("chunk-*/episode_*.parquet"):
            try:
                max_idx = max(max_idx, int(p.stem.split("_")[1]))
            except Exception:
                pass
        return max_idx + 1

    def _discover_global_index(self) -> int:
        # 简化处理：从已有 parquet 行数恢复 global index。
        # 如果没有 pyarrow/pandas，就从 0 开始，不影响单次采集。
        total = 0
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            return 0
        for p in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
            try:
                if pq is not None:
                    total += pq.read_metadata(p).num_rows
                elif pd is not None:
                    total += len(pd.read_parquet(p))
            except Exception:
                pass
        return total

    def episode_chunk(self, episode_index: int) -> int:
        return int(episode_index // self.chunk_size)

    def data_path_for_episode(self, episode_index: int) -> Path:
        chunk = self.episode_chunk(episode_index)
        return self.dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    def video_path_for_episode(self, episode_index: int) -> Path:
        chunk = self.episode_chunk(episode_index)
        return (
            self.dataset_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / f"observation.images.{self.camera_key}"
            / f"episode_{episode_index:06d}.mp4"
        )

    def write_tasks_jsonl(self):
        path = self.dataset_root / "meta" / "tasks.jsonl"
        rows = [
            {"task_index": 0, "task": self.task},
            {"task_index": 1, "task": "valid"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_episode_jsonl(self, episode_index: int, length: int):
        path = self.dataset_root / "meta" / "episodes.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "episode_index": int(episode_index),
                "tasks": [self.task, "valid"],
                "length": int(length),
            }, ensure_ascii=False) + "\n")

    def write_info_json(self):
        total_episodes = self.count_episode_files()
        total_frames = self.global_index
        total_chunks = max(1, math.ceil(max(total_episodes, 1) / max(self.chunk_size, 1)))

        features = {
            f"observation.images.{self.camera_key}": {
                "dtype": "video",
                "shape": [self.resize_height, self.resize_width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(self.fps),
                    "video.codec": self.video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [7],
                "names": list(self.state_names),
            },
            "action": {
                "dtype": "float64",
                "shape": [7],
                "names": list(self.action_names),
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.validity": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        }

        info = {
            "codebase_version": "v2.0",
            "robot_type": self.robot_type,
            "fps": float(self.fps),
            "total_episodes": int(total_episodes),
            "total_frames": int(total_frames),
            "total_tasks": 2,
            "total_videos": int(total_episodes),
            "total_chunks": int(total_chunks),
            "chunks_size": int(self.chunk_size),
            "splits": {"train": f"0:{int(total_episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

        path = self.dataset_root / "meta" / "info.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def write_modality_json(self):
        """
        GR00T-N1.5 训练需要 modality.json。
        这里按单臂 Piper 的 7 维关节状态/动作写。
        后面如果你要双臂，就把 dim 改成 14 或 16。
        """
        modality = {
            "state": {
                "single_arm": {
                    "start": 0,
                    "end": 7,
                }
            },
            "action": {
                "single_arm": {
                    "start": 0,
                    "end": 7,
                }
            },
            "video": {
                self.camera_key: {
                    "original_key": f"observation.images.{self.camera_key}",
                }
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": "annotation.human.action.task_description",
                },
                "human.validity": {
                    "original_key": "annotation.human.validity",
                },
            },
        }
        path = self.dataset_root / "meta" / "modality.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(modality, f, indent=2, ensure_ascii=False)

    def write_stats_json(self):
        path = self.dataset_root / "meta" / "stats.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.global_stats.to_jsonable(), f, indent=2, ensure_ascii=False)

    def count_episode_files(self) -> int:
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            return 0
        return sum(1 for _ in data_dir.glob("chunk-*/episode_*.parquet"))

    def build_episode_arrays(self, episode_index: int, task_index: int):
        ep = self.current_episode
        if ep is None:
            raise RuntimeError("no current episode")
        n = len(ep.image_stamps_ns)
        if n == 0:
            raise RuntimeError("zero frames")

        image_stamps = np.asarray(ep.image_stamps_ns, dtype=np.int64)
        actions = np.stack(ep.actions, axis=0).astype(np.float64)
        states = np.stack(ep.states, axis=0).astype(np.float64)

        timestamps = ((image_stamps - image_stamps[0]).astype(np.float64) / 1e9).astype(np.float64)
        episode_index_arr = np.full(n, int(episode_index), dtype=np.int64)
        index_arr = np.arange(self.global_index, self.global_index + n, dtype=np.int64)
        task_index_arr = np.full(n, int(task_index), dtype=np.int64)

        task_desc_arr = np.zeros(n, dtype=np.int64)
        validity_arr = np.ones(n, dtype=np.int64)
        reward = np.zeros(n, dtype=np.float64)
        done = np.zeros(n, dtype=np.bool_)
        done[-1] = True

        self.global_index += n

        return {
            "observation.state": states,
            "action": actions,
            "timestamp": timestamps,
            "annotation.human.action.task_description": task_desc_arr,
            "task_index": task_index_arr,
            "annotation.human.validity": validity_arr,
            "episode_index": episode_index_arr,
            "index": index_arr,
            "next.reward": reward,
            "next.done": done,
        }

    def write_parquet(self, path: Path, arrays: dict):
        path.parent.mkdir(parents=True, exist_ok=True)

        if pa is not None and pq is not None:
            fields = []
            cols = []
            for name, arr in arrays.items():
                arr = np.asarray(arr)
                if name in ("observation.state", "action"):
                    rows = [np.asarray(row, dtype=np.float64).tolist() for row in arr]
                    typ = pa.list_(pa.float64())
                    col = pa.array(rows, type=typ)
                elif arr.dtype.kind == "f":
                    typ = pa.float64()
                    col = pa.array(arr.astype(np.float64), type=typ)
                elif arr.dtype.kind in ("i", "u"):
                    typ = pa.int64()
                    col = pa.array(arr.astype(np.int64), type=typ)
                elif arr.dtype.kind == "b":
                    typ = pa.bool_()
                    col = pa.array(arr.astype(np.bool_), type=typ)
                else:
                    col = pa.array(arr.tolist())
                    typ = col.type
                fields.append(pa.field(name, typ))
                cols.append(col)

            table = pa.Table.from_arrays(cols, schema=pa.schema(fields))
            pq.write_table(table, path)
            return

        if pd is None:
            raise RuntimeError("Need pyarrow or pandas to write parquet. Install: pip install pyarrow")

        data = {}
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if arr.ndim == 2:
                data[name] = [row.tolist() for row in arr]
            else:
                data[name] = arr.tolist()
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def update_stats(self, arrays: dict):
        for k, v in arrays.items():
            self.global_stats.update(k, np.asarray(v))

    def cleanup_partial(self):
        if self.current_video_writer is not None:
            self.current_video_writer.close()
            self.current_video_writer = None
        if self.current_video_temp_path is not None and self.current_video_temp_path.exists():
            try:
                self.current_video_temp_path.unlink()
            except Exception:
                pass
        self.current_video_temp_path = None
        self.current_episode = None


def main():
    node = PiperMasterSlaveGr00tRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()

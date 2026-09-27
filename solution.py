from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque
import random

import cv2
import numpy as np
import torch
from ultralytics import YOLO


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Official event classes.
CLASSES = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]


# COCO class IDs used by YOLO:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = {2, 3, 5, 7}

# Process every third frame.
# This makes inference significantly faster.
FRAME_STRIDE = 3

# Movement thresholds in normalized image coordinates.
STOP_SPEED = 0.006
CONGESTION_SPEED = 0.012

STOP_MIN_SECONDS = 10.0
CONGESTION_MIN_SECONDS = 5.0

MIN_CONGESTION_VEHICLES = 6

_MODEL = None


def get_model():
    """
    Load YOLO only once.
    """

    global _MODEL

    if _MODEL is None:

        model_path = (
            Path(__file__).resolve().parent
            / "weights"
            / "yolo11n.pt"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights were not found at: {model_path}"
            )

        _MODEL = YOLO(str(model_path))

    return _MODEL


def estimate_speed(history, lookback=1.0):
    """
    Estimate track movement speed using normalized image coordinates.

    history item:
    (time_seconds, x_normalized, y_normalized)
    """

    if len(history) < 2:
        return None

    latest = history[-1]
    earlier = history[0]

    for point in reversed(history):

        if latest[0] - point[0] >= lookback:
            earlier = point
            break

    dt = latest[0] - earlier[0]

    if dt < 0.4:
        return None

    dx = latest[1] - earlier[1]
    dy = latest[2] - earlier[2]

    distance = np.hypot(dx, dy)

    return float(distance / dt)


def merge_events(events, max_gap=0.7):
    """
    Merge overlapping or almost adjacent events
    of the same class.
    """

    if not events:
        return []

    events = sorted(
        events,
        key=lambda event: (event[2], event[0])
    )

    merged = []

    for start, end, label in events:

        if end <= start:
            continue

        if (
            merged
            and merged[-1][2] == label
            and start <= merged[-1][1] + max_gap
        ):

            merged[-1][1] = max(
                merged[-1][1],
                end
            )

        else:

            merged.append(
                [start, end, label]
            )

    merged.sort(key=lambda event: event[0])

    return [
        [
            round(float(start), 3),
            round(float(end), 3),
            label,
        ]
        for start, end, label in merged
        if end > start
    ]


def detect_events(video_path: str) -> list[list]:
    """
    Part A.

    Current MVP detects:

    - stopped_vehicle
    - congestion

    Pipeline:

    video
    -> YOLO
    -> ByteTrack
    -> vehicle trajectories
    -> movement rules
    -> temporal events
    """

    model = get_model()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        fps = 25.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps

    cap.release()

    # Track movement history.
    histories = defaultdict(
        lambda: deque(maxlen=200)
    )

    last_seen = {}

    # Stopped vehicle state.
    stationary_since = {}
    active_stopped = {}

    # Congestion state.
    congestion_candidate = None
    congestion_active = None

    events = []

    processed_frame = 0

    results = model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack.yaml",
        conf=0.30,
        iou=0.50,
        classes=[2, 3, 5, 7],
        vid_stride=FRAME_STRIDE,
        verbose=False,
    )

    for result in results:

        t_sec = (
            processed_frame
            * FRAME_STRIDE
            / fps
        )

        processed_frame += 1

        height, width = result.orig_shape

        current_vehicle_ids = []
        current_speeds = {}

        boxes = result.boxes

        if (
            boxes is not None
            and boxes.id is not None
        ):

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            coordinates = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            for track_id, class_id, box in zip(
                track_ids,
                class_ids,
                coordinates,
            ):

                if class_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                center_x = (
                    ((x1 + x2) / 2)
                    / width
                )

                center_y = (
                    ((y1 + y2) / 2)
                    / height
                )

                histories[track_id].append(
                    (
                        t_sec,
                        float(center_x),
                        float(center_y),
                    )
                )

                last_seen[track_id] = t_sec

                current_vehicle_ids.append(
                    track_id
                )

                speed = estimate_speed(
                    histories[track_id]
                )

                if speed is not None:
                    current_speeds[track_id] = speed

        # -----------------------------------------
        # CONGESTION DETECTION
        # -----------------------------------------

        speeds = list(
            current_speeds.values()
        )

        congestion_now = False

        if (
            len(current_vehicle_ids)
            >= MIN_CONGESTION_VEHICLES
            and len(speeds) >= 3
        ):

            median_speed = float(
                np.median(speeds)
            )

            slow_fraction = (
                sum(
                    speed < CONGESTION_SPEED
                    for speed in speeds
                )
                / len(speeds)
            )

            congestion_now = (
                median_speed
                < CONGESTION_SPEED
                and slow_fraction >= 0.70
            )

        if congestion_now:

            if congestion_candidate is None:
                congestion_candidate = t_sec

            if (
                congestion_active is None
                and
                t_sec - congestion_candidate
                >= CONGESTION_MIN_SECONDS
            ):

                congestion_active = (
                    congestion_candidate
                )

        else:

            if congestion_active is not None:

                events.append(
                    [
                        congestion_active,
                        t_sec,
                        "congestion",
                    ]
                )

            congestion_candidate = None
            congestion_active = None

        # -----------------------------------------
        # STOPPED VEHICLE DETECTION
        # -----------------------------------------

        for track_id in current_vehicle_ids:

            speed = current_speeds.get(
                track_id
            )

            if speed is None:
                continue

            # Avoid calling every vehicle
            # in a traffic jam a stopped vehicle.
            if congestion_now:

                stationary_since.pop(
                    track_id,
                    None
                )

                if track_id in active_stopped:

                    start = active_stopped.pop(
                        track_id
                    )

                    if t_sec > start:

                        events.append(
                            [
                                start,
                                t_sec,
                                "stopped_vehicle",
                            ]
                        )

                continue

            if speed < STOP_SPEED:

                if track_id not in stationary_since:

                    stationary_since[
                        track_id
                    ] = t_sec

                stopped_for = (
                    t_sec
                    - stationary_since[
                        track_id
                    ]
                )

                if (
                    stopped_for
                    >= STOP_MIN_SECONDS
                    and
                    track_id
                    not in active_stopped
                ):

                    active_stopped[
                        track_id
                    ] = stationary_since[
                        track_id
                    ]

            else:

                stationary_since.pop(
                    track_id,
                    None
                )

                if track_id in active_stopped:

                    start = active_stopped.pop(
                        track_id
                    )

                    if t_sec > start:

                        events.append(
                            [
                                start,
                                t_sec,
                                "stopped_vehicle",
                            ]
                        )

        # -----------------------------------------
        # REMOVE TRACKS THAT DISAPPEARED
        # -----------------------------------------

        stale_tracks = [
            track_id
            for track_id, seen_time
            in last_seen.items()
            if t_sec - seen_time > 2.0
        ]

        for track_id in stale_tracks:

            if track_id in active_stopped:

                start = active_stopped.pop(
                    track_id
                )

                end = last_seen[track_id]

                if end > start:

                    events.append(
                        [
                            start,
                            end,
                            "stopped_vehicle",
                        ]
                    )

            stationary_since.pop(
                track_id,
                None
            )

            last_seen.pop(
                track_id,
                None
            )

            histories.pop(
                track_id,
                None
            )

    # -----------------------------------------
    # CLOSE EVENTS AT END OF VIDEO
    # -----------------------------------------

    if (
        congestion_active is not None
        and duration > congestion_active
    ):

        events.append(
            [
                congestion_active,
                duration,
                "congestion",
            ]
        )

    for track_id, start in list(
        active_stopped.items()
    ):

        end = min(
            last_seen.get(
                track_id,
                duration
            ),
            duration,
        )

        if end > start:

            events.append(
                [
                    start,
                    end,
                    "stopped_vehicle",
                ]
            )

    return merge_events(events)


class RiskEstimator:
    """
    Part B is optional.

    For today's MVP we return zero risk,
    which keeps the interface valid.
    """

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.last_score = 0.0

    def step(
        self,
        frame: np.ndarray,
        t_sec: float
    ) -> float:

        return 0.0
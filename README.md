# Traffic Event AI

Computer vision system for detecting traffic events from a fixed CCTV road camera.

## Live Demo

**Public demo:**  
https://traffic-event-ai-uyf744glaujrjwb5sfbild.streamlit.app

**Repository:**  
https://github.com/DescendantOfdaVinci/traffic-event-ai

---

## Problem

The goal is to process an MP4 video from a fixed road camera and return traffic events as temporal segments:

```text
[start_sec, end_sec, label]
```

The official evaluation is performed offline on hidden videos from the same fixed camera viewpoint.

---

## Approach

Our current MVP uses the following pipeline:

```text
Video
→ YOLO11n object detection
→ ByteTrack multi-object tracking
→ vehicle trajectories
→ motion analysis
→ rule-based temporal event detection
```

### Object Detection

YOLO11n detects road vehicles such as:

- cars
- motorcycles
- buses
- trucks

### Tracking

ByteTrack assigns persistent IDs to detected vehicles and allows the system to analyze movement over time.

### Event Detection

The current implementation focuses on two event classes:

- `stopped_vehicle`
- `congestion`

Vehicle trajectories are analyzed in normalized image coordinates.

A vehicle that remains nearly stationary for at least 10 seconds can be classified as `stopped_vehicle`.

Congestion is detected when multiple tracked vehicles remain at very low speeds for a sustained period.

Temporal fragments of the same class are merged before returning the final events.

---

## Accident Anticipation

Part B is optional in the current MVP.

`RiskEstimator` currently returns a risk score of `0.0`.

Future work would use vehicle trajectories and time-to-collision estimates to anticipate accidents.

---

## Model

We use:

- Ultralytics YOLO11n pretrained object detector
- ByteTrack multi-object tracker

The YOLO weights are included locally in:

```text
weights/yolo11n.pt
```

No hosted AI API is used during inference.

The submission is designed to run fully offline.

No custom model training was performed for the current MVP.

---

## Data

Development and testing used the unlabeled CCTV sample videos provided by the hackathon organizers.

The hidden evaluation videos were not accessed.

No additional external training dataset was used by our team for the current MVP.

---

## Installation

Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

---

## Run the Submission

Run the system on a folder of videos:

```bash
python run_submission.py \
  --videos samples \
  --out predictions_samples.json \
  --team "Traffic Event AI"
```

---

## Validate Output

```bash
python evaluate.py \
  --pred predictions_samples.json \
  --validate-only
```

Our current sample output passes the official validator:

```text
0 errors
0 warnings
VALID
```

---

## Current Sample Result

On the provided 127.6-second sample video:

```text
Video duration: 127.6 s
Detected events: 3
Risk samples: 3825
Runtime: 114.4 s
Official time budget: 383 s
Validation: VALID
```

The system therefore runs within the official runtime limit on our development machine.

---

## Public Demo

The Streamlit demo allows a visitor to:

1. upload an MP4 video;
2. run traffic-event detection;
3. view detected events in a table;
4. inspect an event timeline;
5. view the raw prediction output.

For the public demo, short video clips are recommended.

---

## Repository Structure

```text
traffic-event-ai/
├── solution.py
├── run_submission.py
├── evaluate.py
├── requirements.txt
├── predictions_samples.json
├── weights/
│   └── yolo11n.pt
├── web/
│   ├── streamlit_app.py
│   └── requirements.txt
├── examples/
└── README.md
```

---

## Limitations

The current MVP detects only a subset of the 14 official event classes.

Events such as:

- wrong-way driving
- jaywalking
- red-light violations
- illegal turns
- near misses
- accidents

require additional scene-specific calibration or interaction analysis.

Because the evaluation videos use the same fixed camera viewpoint, future versions could define lane polygons, crosswalk regions, expected traffic directions and stop lines for more precise event detection.

---

## Future Work

Planned improvements include:

- lane-region calibration
- wrong-way detection
- pedestrian roadway detection
- crosswalk interaction analysis
- traffic-light state recognition
- illegal-turn detection
- near-miss detection
- time-to-collision estimation
- accident anticipation

---

## Team

### Ziyoda Omonova — Team Captain & Computer Vision

ML pipeline, integration, testing, GitHub and project coordination.

### Dilafruz Tursunpulatova — Research & Documentation

Challenge research, documentation and technical report review.

### Nikol Asriyan — Website & Quality Assurance

Website content review, demo testing and final submission check.
---
## Determinism

The project uses a fixed random seed:

```text
SEED = 42
```

The seed is applied to Python `random`, NumPy and PyTorch.

For CUDA inference, cuDNN benchmarking is disabled and deterministic mode is enabled where supported.

The goal is for repeated runs on the same machine to produce the same predictions up to floating-point noise.

---

## Reproducibility


The inference pipeline uses local model weights and does not require an internet connection during evaluation.

No custom training is performed in the current version.

The organizer-provided `run_submission.py` and `evaluate.py` files are kept unchanged.



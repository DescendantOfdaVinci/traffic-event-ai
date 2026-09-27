import os
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solution import detect_events


st.set_page_config(
    page_title="Traffic Event AI",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Event AI")
st.write(
    "Offline computer vision system for detecting traffic events "
    "from a fixed CCTV road camera."
)

st.header("Team")
st.markdown("""
### Ziyoda Omonova — Team Captain & Computer Vision
ML pipeline, integration, testing, GitHub and project coordination.  
📧 ziyodaomonova@webster.edu

### Dilafruz Tursunpulatova — Research & Documentation
Challenge research, documentation and technical report review.  
📧 dtursunpulatova@webster.edu

### Nikol Asriyan — Website & Quality Assurance
Website content review, demo testing and final submission check.  
📧 nikolasriyan@webster.edu
""")

st.header("Problem")
st.write("""
The system analyzes CCTV road footage and returns detected traffic events
as temporal segments:

`[start_sec, end_sec, label]`
""")

st.header("Approach")
st.markdown("""
### Pipeline

**Video → YOLO11n → ByteTrack → vehicle trajectories → rule-based event detection**

The current MVP uses:

- YOLO11n for vehicle detection
- ByteTrack for object tracking
- motion analysis for temporal event detection

Current implemented event classes:

- `stopped_vehicle`
- `congestion`

The system runs fully offline during evaluation.
""")

st.header("Exploratory Data Analysis")
st.write(
    "The supplied footage comes from a fixed 4K CCTV camera. "
    "The stable viewpoint allows consistent vehicle tracking and "
    "camera-specific motion analysis."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Resolution", "3840 × 2160")
c2.metric("Frame rate", "29.97 FPS")
c3.metric("Duration", "127.6 s")
c4.metric("Frames", "3,825")

st.markdown("""
### Key observations

- The camera viewpoint is fixed, so road geometry remains stable between frames.
- Persistent tracking can therefore be used to estimate vehicle trajectories.
- 4K resolution improves object visibility but increases inference cost.
- Processing every third frame reduces computation while preserving enough temporal information for the current motion-based rules.
- The hidden evaluation uses the same camera viewpoint, so future lane and crosswalk calibration can be camera-specific.
""")

st.header("Live Demo")
st.caption(
    "Demo limit: MP4 files up to 200 MB and approximately 2 minutes."
)
uploaded = st.file_uploader(
    "Upload an MP4 video",
    type=["mp4", "MP4"],
)

if uploaded is not None:

    st.video(uploaded.getvalue())

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = tmp.name

    try:
        cap = cv2.VideoCapture(temp_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = frames / fps if fps else 0

        cap.release()

        st.write(f"Duration: **{duration:.1f} seconds**")

        if duration > 120:
            st.warning(
                "For the public demo, please upload a clip under 2 minutes."
            )

        else:
            if st.button("Detect events", type="primary"):

                with st.spinner("Analyzing video..."):
                    events = detect_events(temp_path)

                st.success("Analysis complete.")

                if not events:
                    st.info(
                        "No events from the current MVP classes were detected."
                    )

                else:
                    df = pd.DataFrame(
                        events,
                        columns=[
                            "start_sec",
                            "end_sec",
                            "label",
                        ],
                    )

                    df["duration"] = (
                        df["end_sec"] - df["start_sec"]
                    )

                    st.subheader("Detected Events")
                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    st.subheader("Event Timeline")

                    fig = go.Figure()

                    for _, row in df.iterrows():

                        fig.add_trace(
                            go.Bar(
                                x=[row["duration"]],
                                y=[row["label"]],
                                base=[row["start_sec"]],
                                orientation="h",
                                name=row["label"],
                            )
                        )

                    fig.update_layout(
                        xaxis_title="Time (seconds)",
                        yaxis_title="Event",
                        showlegend=False,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.subheader("Raw Prediction")
                    st.json(events)

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


st.header("Results")
st.write("""
The current baseline successfully runs through the official challenge
submission harness and produces valid event predictions.

A 30-second test clip was processed in approximately 12 seconds,
well within the challenge runtime limit.
""")

st.write(
    "The public demo successfully detects a stopped vehicle "
    "and visualizes the detected temporal segment on an event timeline."
)

st.header("Limitations")
st.write("""
The current MVP detects only a subset of the official traffic-event classes.

Events such as red-light violations, illegal turns, jaywalking and
wrong-way driving require camera-specific scene calibration.

Accident anticipation is not implemented in the current baseline.
""")

st.header("Technical Report")
st.markdown("""
### What worked

Object detection combined with multi-object tracking provides a practical
way to extract vehicle movement from fixed CCTV footage.

### What needs improvement

Object detection alone cannot determine semantic violations such as
illegal turns or red-light running.

### Next steps

- lane-region calibration
- crosswalk detection
- wrong-way detection
- pedestrian-road interaction analysis
- near-miss detection
- time-to-collision accident anticipation
""")

st.caption(
    "Open-weight models only. No hosted inference APIs are used."
)
st.header("Links")

st.markdown("""
- [GitHub Repository](https://github.com/DescendantOfdaVinci/traffic-event-ai)
- [Model Weights — YOLO11n](https://github.com/DescendantOfdaVinci/traffic-event-ai/blob/main/weights/yolo11n.pt)
- [Sample Predictions](https://github.com/DescendantOfdaVinci/traffic-event-ai/blob/main/predictions_samples.json)
- [Technical Documentation](https://github.com/DescendantOfdaVinci/traffic-event-ai#readme)
""")

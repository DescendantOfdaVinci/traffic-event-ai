import os
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# PROJECT IMPORT
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solution import detect_events


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Traffic Event AI",
    page_icon="🚦",
    layout="wide",
)


# ---------------------------------------------------------
# GLOBAL DESIGN
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* FULL PAGE */
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.18),
                transparent 28%
            ),
            #07152f;
        color: #F8FAFC;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* GENERAL TEXT */
    p, li {
        color: #D7E3F5;
        line-height: 1.7;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    h2 {
        margin-top: 3.3rem !important;
        margin-bottom: 1.4rem !important;
        letter-spacing: -0.03em;
    }

    h3 {
        letter-spacing: -0.02em;
    }

    /* HERO */
    .hero {
        padding: 54px 54px;
        border-radius: 28px;
        background:
            linear-gradient(
                135deg,
                #0C1B3D 0%,
                #16336E 55%,
                #2454B5 100%
            );
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 24px 65px rgba(0,0,0,0.35);
        margin-bottom: 50px;
    }

    .hero-badge {
        color: #8CC8FF;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        margin-bottom: 20px;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 4.4rem;
        line-height: 1.0;
        font-weight: 850;
        letter-spacing: -0.05em;
        margin-bottom: 24px;
    }

    .hero-sub {
        color: #D7E8FF;
        font-size: 1.18rem;
        line-height: 1.7;
        max-width: 800px;
        margin-bottom: 25px;
    }

    .hero-tag {
        display: inline-block;
        color: #E3F1FF;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        padding: 8px 14px;
        margin-right: 8px;
        margin-top: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* BORDER CONTAINERS / TEAM CARDS */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #102447;
        border: 1px solid #294B82 !important;
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }

    /* METRICS */
    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #102447,
                #132B55
            );
        border: 1px solid #2D4F87;
        padding: 22px;
        border-radius: 20px;
        min-height: 125px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.15);
    }

    [data-testid="stMetricLabel"] {
        color: #AFC9EE !important;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2rem !important;
    }

    /* CAPTIONS */
    [data-testid="stCaptionContainer"] {
        color: #9DB6D9 !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: #102447;
        border-radius: 18px;
        padding: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #142D59;
        border: 1px dashed #4978BD;
        border-radius: 14px;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(
            135deg,
            #2563EB,
            #3B82F6
        );
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        font-weight: 750;
        padding: 0.7rem 1.4rem;
    }

    .stButton > button:hover {
        background: #4A8DF7;
        color: #FFFFFF;
        border: none;
    }

    /* LINKS */
    a {
        color: #67B7FF !important;
        text-decoration: none;
    }

    a:hover {
        color: #9DD2FF !important;
        text-decoration: underline;
    }

    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        border: 1px solid #294B82;
        border-radius: 16px;
        overflow: hidden;
    }

    /* ALERTS */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* DIVIDERS */
    hr {
        border-color: #294B82;
    }

    /* SMALL SCREEN */
    @media (max-width: 800px) {
        .hero {
            padding: 35px 28px;
        }

        .hero-title {
            font-size: 3rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

hero_html = (
    '<div class="hero">'
    '<div class="hero-badge">'
    'WIUT HACKATHON 2026 · COMPUTER VISION'
    '</div>'
    '<div class="hero-title">Traffic Event AI</div>'
    '<div class="hero-sub">'
    'Detecting traffic events from fixed CCTV footage using '
    'YOLO11n, ByteTrack and trajectory-based reasoning.'
    '</div>'
    '<span class="hero-tag">YOLO11n</span>'
    '<span class="hero-tag">ByteTrack</span>'
    '<span class="hero-tag">Offline inference</span>'
    '<span class="hero-tag">Fixed-camera analytics</span>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# TEAM
# ---------------------------------------------------------

st.header("Team")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("Ziyoda Omonova")
        st.caption("TEAM CAPTAIN · COMPUTER VISION")
        st.write(
            "ML pipeline, integration, testing, "
            "GitHub and project coordination."
        )
        st.markdown(
            "✉️ [ziyodaomonova@webster.edu]"
            "(mailto:ziyodaomonova@webster.edu)"
        )

with col2:
    with st.container(border=True):
        st.subheader("Dilafruz Tursunpulatova")
        st.caption("RESEARCH · DOCUMENTATION")
        st.write(
            "Challenge research, documentation "
            "and technical report review."
        )
        st.markdown(
            "✉️ [dtursunpulatova@webster.edu]"
            "(mailto:dtursunpulatova@webster.edu)"
        )

with col3:
    with st.container(border=True):
        st.subheader("Nikol Asriyan")
        st.caption("WEBSITE · QUALITY ASSURANCE")
        st.write(
            "Website content review, demo testing "
            "and final submission check."
        )
        st.markdown(
            "✉️ [nikolasriyan@webster.edu]"
            "(mailto:nikolasriyan@webster.edu)"
        )


# ---------------------------------------------------------
# PROBLEM
# ---------------------------------------------------------

st.header("Problem")

st.write(
    """
    Traffic monitoring cameras generate large amounts of footage,
    but manually reviewing it is slow and difficult to scale.

    Our system analyzes footage from a fixed CCTV road camera and
    returns detected traffic events as temporal segments:
    """
)

st.code(
    "[start_sec, end_sec, label]",
    language="text",
)


# ---------------------------------------------------------
# APPROACH
# ---------------------------------------------------------

st.header("Approach")

st.subheader("Pipeline")

st.markdown(
    """
    **Video → YOLO11n → ByteTrack → vehicle trajectories
    → motion analysis → temporal events**
    """
)

a1, a2, a3 = st.columns(3)

with a1:
    with st.container(border=True):
        st.subheader("01 · Detect")
        st.write(
            "YOLO11n detects cars, motorcycles, buses "
            "and trucks in each sampled frame."
        )

with a2:
    with st.container(border=True):
        st.subheader("02 · Track")
        st.write(
            "ByteTrack assigns persistent IDs so vehicle "
            "movement can be analyzed over time."
        )

with a3:
    with st.container(border=True):
        st.subheader("03 · Reason")
        st.write(
            "Trajectory-based rules convert motion patterns "
            "into temporal traffic-event segments."
        )

st.markdown(
    """
    **Currently implemented event classes**

    - `stopped_vehicle`
    - `congestion`

    The evaluation pipeline runs fully offline using local model weights.
    """
)


# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------

st.header("Exploratory Data Analysis")

st.write(
    "The supplied footage comes from a fixed 4K CCTV camera. "
    "The stable viewpoint allows consistent vehicle tracking "
    "and camera-specific motion analysis."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Resolution", "4K")
c2.metric("Frame rate", "29.97 FPS")
c3.metric("Duration", "127.6 s")
c4.metric("Frames", "3,825")

st.caption("Native resolution: 3840 × 2160")

st.subheader("Key observations")

st.markdown(
    """
    - The camera viewpoint is fixed, so road geometry remains stable between frames.
    - Persistent tracking can therefore be used to estimate vehicle trajectories.
    - 4K resolution improves object visibility but increases inference cost.
    - Processing every third frame reduces computation while preserving enough temporal information for the current motion-based rules.
    - The hidden evaluation uses the same camera viewpoint, making future lane and crosswalk calibration camera-specific.
    """
)


# ---------------------------------------------------------
# LIVE DEMO
# ---------------------------------------------------------

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
        suffix=".mp4",
    ) as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = tmp.name

    try:

        cap = cv2.VideoCapture(temp_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = (
            frames / fps
            if fps and fps > 0
            else 0
        )

        cap.release()

        v1, v2 = st.columns(2)

        v1.metric(
            "Uploaded duration",
            f"{duration:.1f} s",
        )

        v2.metric(
            "Demo limit",
            "120 s",
        )

        if duration > 120:

            st.warning(
                "For the public demo, please upload "
                "a clip under approximately 2 minutes."
            )

        else:

            if st.button(
                "Detect events",
                type="primary",
            ):

                with st.spinner(
                    "Analyzing traffic footage..."
                ):
                    events = detect_events(temp_path)

                st.success(
                    "Analysis complete."
                )

                if not events:

                    st.info(
                        "No events from the current MVP "
                        "classes were detected."
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
                        df["end_sec"]
                        - df["start_sec"]
                    )

                    r1, r2 = st.columns(2)

                    r1.metric(
                        "Detected events",
                        len(events),
                    )

                    r2.metric(
                        "Event classes",
                        df["label"].nunique(),
                    )

                    st.subheader(
                        "Detected Events"
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader(
                        "Event Timeline"
                    )

                    fig = go.Figure()

                    for _, row in df.iterrows():

                        fig.add_trace(
                            go.Bar(
                                x=[row["duration"]],
                                y=[row["label"]],
                                base=[
                                    row["start_sec"]
                                ],
                                orientation="h",
                                name=row["label"],
                                hovertemplate=(
                                    f"<b>{row['label']}</b><br>"
                                    f"Start: {row['start_sec']:.2f}s<br>"
                                    f"End: {row['end_sec']:.2f}s"
                                    "<extra></extra>"
                                ),
                            )
                        )

                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Event",
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(
                            l=30,
                            r=30,
                            t=30,
                            b=30,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.subheader(
                        "Raw Prediction"
                    )

                    st.json(events)

    finally:

        try:
            os.remove(temp_path)
        except OSError:
            pass


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

st.header("Results")

r1, r2, r3 = st.columns(3)

r1.metric(
    "Sample events",
    "3",
)

r2.metric(
    "Runtime",
    "114.4 s",
)

r3.metric(
    "Validator",
    "VALID",
)

st.write(
    """
    On the 127.6-second development sample, the system detected
    three temporal events and completed inference in 114.4 seconds.

    The official runtime budget for the same clip was 383 seconds,
    so the current pipeline remained comfortably within the required limit.
    """
)

st.write(
    "The public demo also successfully detects a stopped vehicle "
    "and visualizes its temporal segment on an interactive timeline."
)


# ---------------------------------------------------------
# LIMITATIONS
# ---------------------------------------------------------

st.header("Limitations")

st.write(
    """
    The current MVP intentionally focuses on a small number of
    motion-based event classes.

    Events such as red-light violations, illegal turns,
    jaywalking and wrong-way driving require additional
    camera-specific scene calibration.

    Accident anticipation is not implemented in the current baseline.
    """
)


# ---------------------------------------------------------
# TECHNICAL REPORT
# ---------------------------------------------------------

st.header("Technical Report")

report1, report2 = st.columns(2)

with report1:
    with st.container(border=True):

        st.subheader("What worked")

        st.write(
            """
            Combining object detection with multi-object tracking
            provides a practical way to extract vehicle movement
            from fixed CCTV footage.

            The fixed camera viewpoint makes trajectory-based
            reasoning especially useful.
            """
        )

with report2:
    with st.container(border=True):

        st.subheader("What needs improvement")

        st.write(
            """
            Generic object detection alone cannot reliably identify
            semantic violations such as illegal turns or red-light
            running.

            These tasks require scene-specific road geometry and
            interaction analysis.
            """
        )

st.subheader("Next steps")

st.markdown(
    """
    - lane-region calibration
    - crosswalk detection
    - wrong-way detection
    - pedestrian-road interaction analysis
    - near-miss detection
    - time-to-collision accident anticipation
    """
)

st.caption(
    "Open-weight models only. "
    "No hosted inference APIs are used."
)


# ---------------------------------------------------------
# LINKS
# ---------------------------------------------------------

st.header("Links")

l1, l2 = st.columns(2)

with l1:
    with st.container(border=True):

        st.subheader("Code & Models")

        st.markdown(
            """
            - [GitHub Repository](https://github.com/DescendantOfdaVinci/traffic-event-ai)
            - [YOLO11n Model Weights](https://github.com/DescendantOfdaVinci/traffic-event-ai/blob/main/weights/yolo11n.pt)
            """
        )

with l2:
    with st.container(border=True):

        st.subheader("Results & Documentation")

        st.markdown(
            """
            - [Sample Predictions](https://github.com/DescendantOfdaVinci/traffic-event-ai/blob/main/predictions_samples.json)
            - [Technical Documentation](https://github.com/DescendantOfdaVinci/traffic-event-ai#readme)
            """
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Traffic Event AI · WIUT Hackathon 2026 · Computer Vision Track"
)
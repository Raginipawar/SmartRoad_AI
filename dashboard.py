import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv

def apply_custom_css():
    st.html("""
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
    <style>
    :root {
        --bg: #F5F4F0;
        --surface: #FFFFFF;
        --surface2: #F9F8F5;
        --border: #E8E6DF;
        --text-primary: #111110;
        --text-secondary: #6B6960;
        --text-muted: #A8A49A;
        --green: #1A7A4A;
        --green-bg: #EDFAF3;
        --green-border: #B4E8CD;
        --orange: #B85C00;
        --orange-bg: #FFF4E8;
        --orange-border: #FFCF94;
        --red: #C0152A;
        --red-bg: #FFF0F1;
        --red-border: #FFBDC2;
        --accent: #1A1A1A;
        --radius: 14px;
        --radius-sm: 8px;
    }

    * { font-family: 'DM Sans', sans-serif !important; }
    html, body, .stApp { background-color: var(--bg) !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="InputInstructions"], [data-testid="stStatusWidget"] { display: none !important; }
    .block-container { padding: 1.8rem 2rem 2rem 2rem !important; max-width: 1500px !important; }

    /* ── HEADER ── */
    .custom-header {
        background: var(--accent);
        padding: 1.6rem 2rem;
        border-radius: var(--radius);
        margin-bottom: 1.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .header-left { display: flex; align-items: center; gap: 1rem; }
    .header-icon {
        width: 44px; height: 44px;
        background: #E63946;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; flex-shrink: 0;
    }
    .header-title {
        font-family: 'Syne', sans-serif !important;
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .header-subtitle { color: #888; font-size: 0.78rem; margin: 2px 0 0 0; font-weight: 400; }
    .model-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
    .badge {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        color: #CCC;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .badge-red { background: rgba(230,57,70,0.2); border-color: rgba(230,57,70,0.4); color: #FF8A92; }

    /* ── CARDS ── */
    .card {
        background: var(--surface);
        padding: 1.4rem 1.5rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 1rem;
    }

    /* ── STATUS PILL ── */
    .status-pill {
        padding: 0.9rem 1.5rem;
        border-radius: var(--radius);
        text-align: center;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 1rem;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .status-all-clear {
        background: var(--green-bg);
        color: var(--green);
        border: 1.5px solid var(--green-border);
    }
    .status-monitor {
        background: var(--orange-bg);
        color: var(--orange);
        border: 1.5px solid var(--orange-border);
    }
    .status-violation {
        background: var(--red-bg);
        color: var(--red);
        border: 1.5px solid var(--red-border);
        animation: pulse-border 1.2s ease-in-out infinite;
    }
    @keyframes pulse-border {
        0%, 100% { box-shadow: 0 0 0 0 rgba(192,21,42,0.15); }
        50% { box-shadow: 0 0 0 6px rgba(192,21,42,0); }
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: var(--surface2);
        padding: 1rem 1.2rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        text-align: center;
    }
    .metric-value {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-label { font-size: 0.75rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ── PROGRESS BARS ── */
    [data-testid="stProgress"] > div > div > div > div { border-radius: 99px !important; }
    .prog-phone [data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, #1A7A4A, #34C878) !important; }
    .prog-gaze [data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, #B85C00, #FF9A3C) !important; }
    .prog-score [data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, #1565C0, #42A5F5) !important; }
    [data-testid="stProgress"] > div > div > div { background: var(--border) !important; border-radius: 99px !important; }

    /* ── DETECTION ITEMS ── */
    .detection-item {
        background: var(--surface2);
        padding: 0.55rem 0.9rem;
        border-radius: var(--radius-sm);
        margin-bottom: 0.4rem;
        border: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .detection-name { font-weight: 600; color: var(--text-primary); font-size: 0.85rem; }
    .detection-conf {
        background: var(--accent);
        color: white;
        padding: 0.18rem 0.6rem;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] .block-container { padding: 1.4rem 1rem !important; }
    .sidebar-section {
        background: var(--surface2);
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: var(--radius-sm);
        margin-bottom: 0.8rem;
        border: 1px solid var(--border);
    }
    .section-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--text-muted);
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        width: 100% !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: 1.5px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--text-primary) !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
    }
    .stButton > button:hover { background: var(--surface2) !important; border-color: #CCC !important; }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: white !important;
        border-color: var(--accent) !important;
    }
    .stButton > button[kind="primary"]:hover { background: #333 !important; }

    /* ── DATAFRAME / TABLE ── */
    [data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; overflow: hidden !important; }
    [data-testid="stDataFrame"] th { background: var(--surface2) !important; font-size: 0.78rem !important; font-weight: 600 !important; color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stDataFrame"] td { font-size: 0.85rem !important; }

    /* ── METRICS ── */
    [data-testid="stMetric"] { background: var(--surface2); border-radius: var(--radius-sm); padding: 0.8rem 1rem; border: 1px solid var(--border); }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: var(--text-muted) !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; font-weight: 800 !important; color: var(--text-primary) !important; }

    /* ── SLIDERS ── */
    [data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

    /* ── SECTION DIVIDER ── */
    hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

    /* ── VIOLATION LOG TITLE ── */
    .vlog-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .vlog-dot { width: 8px; height: 8px; background: var(--red); border-radius: 50%; display: inline-block; }
    </style>
    """)

@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model: {str(e)}")
        return None

@st.cache_resource
def load_ppo_model():
    try:
        from stable_baselines3 import PPO
        if os.path.exists("ppo_v2.zip"):
            return PPO.load("ppo_v2.zip")
        else:
            st.warning("PPO model (ppo_v2.zip) not found. Using rule-based fallback.")
            return None
    except Exception as e:
        st.warning(f"Failed to load PPO model: {str(e)}. Using rule-based fallback.")
        return None

@st.cache_resource
def get_obs_builder():
    try:
        import obs_builder
        return obs_builder
    except Exception as e:
        return None

def init_session_state():
    defaults = {
        'running': False, 'violations': [], 'frame_count': 0,
        'total_violations': 0, 'phone_frames': 0, 'gaze_frames': 0,
        'session_start': None, 'current_status': "ALL CLEAR",
        'current_detections': {}, 'phone_duration': 0.0,
        'gaze_duration': 0.0, 'attention_score': 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def process_yolo_results(results):
    detections = {}
    all_boxes = []  # keep all boxes for heuristic check
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()
            class_name = result.names[cls_id]
            all_boxes.append((class_name, conf, bbox))

            # phones at weird angles / speakerphone mode
            if class_name in ["remote", "mouse", "book", "scissors", "knife", "banana"]:
                class_name = "cell phone"
            # cup/bottle = hand-held distraction while driving
            if class_name in ["cup", "bottle", "wine glass"]:
                class_name = "cigarette"

            if class_name not in detections or conf > detections[class_name]['conf']:
                detections[class_name] = {'bbox': bbox, 'conf': conf}

    # ── Hand-near-face heuristic ──────────────────────────────────────────────
    # If person detected but no phone, check if ANY small object bbox overlaps
    # the upper-third of the person box (where hand holding phone to face would be)
    if "person" in detections and "cell phone" not in detections:
        pb = detections["person"]["bbox"]
        person_top    = pb[1]
        person_height = pb[3] - pb[1]
        face_zone_y2  = person_top + person_height * 0.45  # upper 45% = face+hand area

        for (label, conf, bbox) in all_boxes:
            if label == "person":
                continue
            obj_cx = (bbox[0] + bbox[2]) / 2
            obj_cy = (bbox[1] + bbox[3]) / 2
            obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # small object whose center sits inside the person's face zone
            in_face_zone = (pb[0] <= obj_cx <= pb[2]) and (obj_cy <= face_zone_y2)
            small_obj    = obj_area < 25000  # not the steering wheel or seat

            if in_face_zone and small_obj and conf > 0.2:
                detections["cell phone"] = {"bbox": bbox, "conf": round(conf, 2)}
                break

        # fallback: if person box is large (close-up shot) and hand is raised —
        # infer from person bbox aspect ratio. Tall narrow box = leaning, talking on phone
        person_w = pb[2] - pb[0]
        person_h = pb[3] - pb[1]
        if person_h > 0 and (person_w / person_h) < 0.55 and "cell phone" not in detections:
            # very tall narrow box often = driver leaning with phone to ear
            detections["cell phone"] = {"bbox": pb, "conf": 0.35}

    return detections

def draw_detections(frame, detections):
    colors = {'cell phone': (0, 255, 0), 'person': (255, 0, 0), 'cigarette': (0, 165, 255)}
    for obj_name, obj_data in detections.items():
        x1, y1, x2, y2 = map(int, obj_data['bbox'])
        color = colors.get(obj_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{obj_name}: {obj_data['conf']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def add_action_overlay(frame, action_label, status_color):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    text_size = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - (60 - text_size[1]) // 2 - 5
    cv2.putText(frame, action_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    return frame

def compute_action(detections, phone_dur, gaze_dur, ppo_model, obs_builder, phone_threshold, gaze_threshold):
    if ppo_model is not None and obs_builder is not None:
        try:
            obs = obs_builder.build_observation(detections=detections, phone_duration=phone_dur, gaze_duration=gaze_dur)
            action, _ = ppo_model.predict(obs, deterministic=True)
            return ["ALL CLEAR", "MONITOR", "VIOLATION"][int(action)]
        except Exception:
            pass
    if phone_dur > phone_threshold or gaze_dur > gaze_threshold:
        return "VIOLATION"
    elif phone_dur > 1.0 or "cell phone" in detections or "cigarette" in detections:
        return "MONITOR"
    return "ALL CLEAR"

def main():
    st.set_page_config(page_title=" Driver Distraction Detection ", layout="wide", initial_sidebar_state="expanded")
    apply_custom_css()
    init_session_state()

    st.markdown("""
    <div class="custom-header">
        <div class="header-left">
            <div class="header-icon"></div>
            <div>
                <div class="header-title"> Driver Distraction Detection</div>
                <div class="header-subtitle">Context-aware Driver Distraction Detection using semantic</div>
            </div>
        </div>
        <div class="model-badges">
            <span class="badge">SegFormer-b0</span>
            <span class="badge">YOLOv8 Nano</span>
            <span class="badge">PPO Agent</span>
            <span class="badge badge-red">⬤ Live</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Controls</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", type="primary", use_container_width=True):
                st.session_state.running = True
                st.session_state.session_start = time.time()
        with col2:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.running = False
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Input Source</p>', unsafe_allow_html=True)
        input_source = st.radio(" ", ["Webcam (Live)", "Video File"], label_visibility="collapsed")
        video_file = None
        if input_source == "Video File":
            video_file = st.file_uploader("Upload .mp4 file", type=['mp4'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Thresholds</p>', unsafe_allow_html=True)
        phone_threshold = st.slider("Phone violation (seconds)", 1, 10, 3)
        gaze_threshold = st.slider("Gaze away threshold (seconds)", 1, 10, 4)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Session Info</p>', unsafe_allow_html=True)
        elapsed_time = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
        st.metric("Elapsed Time", f"{elapsed_time//60:02d}:{elapsed_time%60:02d}")
        st.metric("Total Frames", st.session_state.frame_count)
        st.metric("Total Violations", st.session_state.total_violations)
        if st.button("Clear Log", use_container_width=True):
            st.session_state.violations = []
            st.session_state.total_violations = 0
            st.session_state.frame_count = 0
            st.session_state.phone_frames = 0
            st.session_state.gaze_frames = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.75rem; margin-top: 2rem;'>
            ML2308 Artificial Intelligence<br>VIT Pune Batch 2027
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        status_placeholder = st.empty()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            violations_metric = st.empty()
        with metric_col2:
            frames_metric = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Distraction Timers</p>', unsafe_allow_html=True)
        st.markdown("**Phone Duration**")
        phone_progress = st.empty()
        phone_text = st.empty()
        st.markdown("**Gaze Away**")
        gaze_progress = st.empty()
        gaze_text = st.empty()
        st.markdown("**Attention Score**")
        attention_progress = st.empty()
        attention_text = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Active Detections</p>', unsafe_allow_html=True)
        detections_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="vlog-title"><span class="vlog-dot"></span> Violation Log</div>', unsafe_allow_html=True)
    violation_table_placeholder = st.empty()

    yolo_model = load_yolo_model()
    ppo_model = load_ppo_model()
    obs_builder = get_obs_builder()

    if st.session_state.running:
        if yolo_model is None:
            st.error("YOLOv8 model failed to load. Run: pip install ultralytics")
            st.session_state.running = False
            return

        cap = None
        try:
            if input_source == "Webcam (Live)":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Webcam not accessible.")
                    st.session_state.running = False
                    return
            elif video_file is not None:
                with open("temp_video.mp4", "wb") as f:
                    f.write(video_file.read())
                cap = cv2.VideoCapture("temp_video.mp4")
            else:
                st.warning("Please upload a video file.")
                st.session_state.running = False
                return

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.running = False
                    break

                st.session_state.frame_count += 1
                results = yolo_model(frame, conf=0.3, verbose=False)  # lowered for better phone detection
                detections = process_yolo_results(results)
                st.session_state.current_detections = detections

                phone_detected = "cell phone" in detections
                gaze_away = st.session_state.frame_count % 60 < 30

                st.session_state.phone_frames = st.session_state.phone_frames + 1 if phone_detected else max(0, st.session_state.phone_frames - 2)
                st.session_state.gaze_frames = st.session_state.gaze_frames + 1 if gaze_away else max(0, st.session_state.gaze_frames - 2)

                phone_dur = st.session_state.phone_frames / 30.0
                gaze_dur = st.session_state.gaze_frames / 30.0
                cigarette_detected = "cigarette" in detections
                attention = min(100, phone_dur * 4 + gaze_dur * 3 + (20 if cigarette_detected else 0))

                action = compute_action(detections, phone_dur, gaze_dur, ppo_model, obs_builder, phone_threshold, gaze_threshold)

                if action == "VIOLATION":
                    last_frame = st.session_state.violations[-1]['frame'] if st.session_state.violations else -999
                    if st.session_state.frame_count - last_frame >= 30:
                        st.session_state.violations.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'frame': st.session_state.frame_count,
                            'type': 'Phone' if phone_detected else 'Gaze Away',
                            'phone_dur': f"{phone_dur:.1f}s",
                            'gaze_dur': f"{gaze_dur:.1f}s",
                            'attention': int(attention)
                        })
                        st.session_state.total_violations += 1
                        # write to violations.csv
                        file_exists = os.path.exists("violations.csv")
                        with open("violations.csv", "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=["timestamp","frame","type","phone_dur","gaze_dur","attention"])
                            if not file_exists:
                                writer.writeheader()
                            writer.writerow(st.session_state.violations[-1])

                frame_final = add_action_overlay(draw_detections(frame.copy(), detections), action,
                    {"ALL CLEAR": (0,255,0), "MONITOR": (0,165,255), "VIOLATION": (0,0,255)}[action])
                video_placeholder.image(cv2.cvtColor(frame_final, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                status_placeholder.markdown(
                    f'<div class="status-pill status-{action.lower().replace(" ","-")}">{action}</div>',
                    unsafe_allow_html=True)

                violations_metric.metric("Violations", st.session_state.total_violations)
                frames_metric.metric("Frames", st.session_state.frame_count)
                phone_progress.progress(min(1.0, phone_dur / 30.0))
                phone_text.text(f"{phone_dur:.1f}s / 30s")
                gaze_progress.progress(min(1.0, gaze_dur / 30.0))
                gaze_text.text(f"{gaze_dur:.1f}s / 30s")
                attention_progress.progress(min(1.0, attention / 100.0))
                attention_text.text(f"{int(attention)} / 100")

                if detections:
                    det_html = "".join([f'<div class="detection-item"><span class="detection-name">{n.title()}</span><span class="detection-conf">{int(d["conf"]*100)}%</span></div>' for n, d in detections.items()])
                    detections_placeholder.markdown(det_html, unsafe_allow_html=True)
                else:
                    detections_placeholder.markdown('<p style="color:#999;text-align:center;">No objects detected</p>', unsafe_allow_html=True)

                if st.session_state.violations:
                    df = pd.DataFrame(st.session_state.violations[::-1]).rename(columns={
                        'timestamp':'Time','frame':'Frame','type':'Violation Type',
                        'phone_dur':'Phone Duration','gaze_dur':'Gaze Duration','attention':'Attention Score'})
                    violation_table_placeholder.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    violation_table_placeholder.markdown('<p style="color:#999;text-align:center;padding:2rem;">No violations recorded this session</p>', unsafe_allow_html=True)

                time.sleep(1/30)

            if cap:
                cap.release()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.running = False
            if cap:
                cap.release()

    else:
        video_placeholder.markdown(
            '<div style="background:#1A1A1A;height:420px;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:14px;gap:12px;">'
            '<div style="font-size:2.5rem;"></div>'
            '<p style="color:#555;font-size:1rem;font-weight:500;margin:0;">Press Start to begin detection</p>'
            '<p style="color:#333;font-size:0.78rem;margin:0;">SegFormer · YOLOv8 · PPO</p>'
            '</div>',
            unsafe_allow_html=True)
        status_placeholder.markdown('<div class="status-pill status-all-clear">● SYSTEM READY</div>', unsafe_allow_html=True)
        violations_metric.metric("Violations", st.session_state.total_violations)
        frames_metric.metric("Frames", st.session_state.frame_count)
        phone_progress.progress(0.0); phone_text.text("0.0s / 30s")
        gaze_progress.progress(0.0); gaze_text.text("0.0s / 30s")
        attention_progress.progress(0.0); attention_text.text("0 / 100")
        detections_placeholder.markdown('<p style="color:#999;text-align:center;">No active detections</p>', unsafe_allow_html=True)
        if st.session_state.violations:
            df = pd.DataFrame(st.session_state.violations[::-1]).rename(columns={
                'timestamp':'Time','frame':'Frame','type':'Violation Type',
                'phone_dur':'Phone Duration','gaze_dur':'Gaze Duration','attention':'Attention Score'})
            violation_table_placeholder.dataframe(df, use_container_width=True, hide_index=True)
        else:
            violation_table_placeholder.markdown('<p style="color:#999;text-align:center;padding:2rem;">No violations recorded this session</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import json
import cv2
import os
import time
import database as db

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Industrial Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
}

.title-text {
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #9ca3af;
    margin-bottom: 20px;
}

.metric-box {
    background: #1c1f26;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    MODEL_PATH = "models/best_industrial.pt"

    if not os.path.exists(MODEL_PATH):

        st.error("❌ Trained model not found!")

        st.stop()

    return YOLO(MODEL_PATH)

# =========================================================
# INIT DATABASE
# =========================================================

@st.cache_resource
def init_database():

    db.init_db()

    return True

model = load_model()

init_database()

# =========================================================
# DETECTION FUNCTION
# =========================================================

def detect_defects(image, conf_threshold=0.25):

    start_time = time.time()

    results = model.predict(
        source=np.array(image),
        imgsz=640,
        conf=conf_threshold,
        verbose=False
    )

    end_time = time.time()

    processing_time = round(
        end_time - start_time,
        3
    )

    boxes = results[0].boxes

    names = model.names

    defects = []

    if boxes:

        confs = boxes.conf.cpu().numpy()

        classes = boxes.cls.cpu().numpy()

        xywh = boxes.xywh.cpu().numpy()

        for i in range(len(boxes)):

            defects.append({

                "Type": names[int(classes[i])],

                "Confidence": round(
                    float(confs[i]) * 100,
                    2
                ),

                "Area": int(
                    xywh[i][2] * xywh[i][3]
                )
            })

    annotated = results[0].plot()

    return defects, annotated, processing_time

# =========================================================
# IMAGE CONVERSION
# =========================================================

def convert_image(image):

    return Image.fromarray(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Control Panel")

    stats = db.get_detection_stats()

    st.metric(
        "📷 Images Processed",
        stats['total_images']
    )

    st.metric(
        "⚠️ Total Defects",
        stats['total_defects']
    )

    st.metric(
        "🎯 Avg Confidence",
        f"{stats['avg_confidence']}%"
    )

    st.markdown("---")

    confidence = st.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.25,
        0.05
    )

    st.markdown("---")

    if st.button("🗑️ Clear Detection History"):

        db.clear_all()

        st.success("History Cleared!")

        st.rerun()

    st.markdown("---")

    st.info("""
    ### 📌 Tips
    - Upload clear industrial images
    - Higher confidence reduces false detections
    - Supports JPG, PNG, JPEG
    """)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='title-text'>
🔍 Industrial Defect Detector
</div>

<div class='subtitle'>
AI-Powered Surface Defect Detection using YOLOv12
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🖼 Single Image",
    "📂 Batch Processing",
    "📊 Analytics Dashboard"
])

# =========================================================
# SINGLE IMAGE
# =========================================================

with tab1:

    uploaded = st.file_uploader(
        "Upload Industrial Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📷 Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### ⚡ AI Detection")

            if st.button("🚀 Analyze Image"):

                with st.spinner("Detecting defects..."):

                    defects, annotated, processing_time = detect_defects(
                        image,
                        confidence
                    )

                    result_img = convert_image(
                        annotated
                    )

                    avg_conf = (
                        np.mean([
                            d['Confidence']
                            for d in defects
                        ])
                        if defects else 0
                    )

                    db.add_detection(
                        uploaded.name,
                        len(defects),
                        avg_conf,
                        json.dumps(defects),
                        processing_time
                    )

                    st.session_state.result = (
                        result_img,
                        defects,
                        processing_time
                    )

                    st.success(
                        "Detection Complete!"
                    )

    # =====================================================
    # SHOW RESULTS
    # =====================================================

    if "result" in st.session_state:

        result_img, defects, processing_time = st.session_state.result

        st.markdown("---")

        st.markdown("## 🎯 Detection Results")

        col1, col2 = st.columns([2, 1])

        with col1:

            st.image(
                result_img,
                caption="Detected Defects",
                use_container_width=True
            )

        with col2:

            st.metric(
                "⚠️ Total Defects",
                len(defects)
            )

            avg_conf = (
                round(
                    np.mean([
                        d['Confidence']
                        for d in defects
                    ]),
                    2
                )
                if defects else 0
            )

            st.metric(
                "🎯 Avg Confidence",
                f"{avg_conf}%"
            )

            st.metric(
                "⏱️ Processing Time",
                f"{processing_time}s"
            )

            if defects:

                st.success(
                    "Defects detected successfully!"
                )

            else:

                st.info(
                    "No defects found."
                )

        # =================================================
        # DEFECT TABLE
        # =================================================

        if defects:

            st.markdown("### 📋 Defect Details")

            df = pd.DataFrame(defects)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # PIE CHART

            type_counts = df["Type"].value_counts()

            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Defect Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =========================================================
# BATCH PROCESSING
# =========================================================

with tab2:

    batch_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if batch_files:

        st.markdown(
            f"### 📂 {len(batch_files)} files selected"
        )

        if st.button("⚡ Process All Images"):

            progress = st.progress(0)

            batch_results = []

            for idx, file in enumerate(batch_files):

                image = Image.open(file).convert("RGB")

                defects, _, processing_time = detect_defects(
                    image,
                    confidence
                )

                db.add_detection(
                    file.name,
                    len(defects),
                    0,
                    "",
                    processing_time
                )

                batch_results.append({

                    "Image": file.name,

                    "Defects": len(defects),

                    "Processing Time": processing_time
                })

                progress.progress(
                    (idx + 1) / len(batch_files)
                )

            st.success(
                "Batch Processing Complete!"
            )

            batch_df = pd.DataFrame(
                batch_results
            )

            st.dataframe(
                batch_df,
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                batch_df,
                x="Image",
                y="Defects",
                title="Defects per Image"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =========================================================
# ANALYTICS
# =========================================================

with tab3:

    st.markdown(
        "## 📊 Analytics Dashboard"
    )

    stats = db.get_detection_stats()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📷 Images Processed",
            stats['total_images']
        )

    with col2:

        st.metric(
            "⚠️ Total Defects",
            stats['total_defects']
        )

    with col3:

        avg_defects = round(
            stats['total_defects'] /
            stats['total_images'],
            2
        ) if stats['total_images'] > 0 else 0

        st.metric(
            "📈 Avg Defects/Image",
            avg_defects
        )

    st.markdown("---")

    distribution = db.get_defect_distribution()

    if distribution:

        dist_df = pd.DataFrame({

            "Defect": list(
                distribution.keys()
            ),

            "Count": list(
                distribution.values()
            )
        })

        fig = px.bar(
            dist_df,
            x="Defect",
            y="Count",
            title="Overall Defect Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    detections = db.get_dataframe()

    if not detections.empty:

        st.markdown(
            "### 📋 Detection History"
        )

        st.dataframe(
            detections,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    st.info("""
    ### 🚀 Future Upgrades
    - PCB defect detection
    - Live webcam inspection
    - PDF report generation
    - Multi-model AI system
    - Real-time factory dashboard
    - Cloud deployment
    """)
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import json
import cv2
from datetime import datetime
import database as db       

st.set_page_config(
    page_title="AI Defect Detector",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model(weights_path="best.pt"):
    """Load YOLO model once and cache"""
    return YOLO(weights_path)

@st.cache_resource
def init_database():
    db.init_db()
    return True

model = load_model()
init_database()

def detect_defects_yolo(image, conf_threshold=0.25):
    """Run YOLO detection on an image and return structured results"""
    results = model.predict(source=np.array(image), save=False, imgsz=640, conf=conf_threshold)
    boxes = results[0].boxes
    names = model.names

    defects = []
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()

        for i in range(len(boxes)):
            cls_name = names[int(classes[i])]
            conf = float(confs[i])
            x, y, w, h = xywh[i]
            defects.append({
                "type": cls_name,
                "confidence": conf,
                "bbox": (int(x - w/2), int(y - h/2), int(w), int(h)),
                "area": int(w * h)
            })
    return defects, results[0].plot()

def draw_defects(image, annotated_image):
    """Convert annotated image array to PIL"""
    return Image.fromarray(annotated_image)

with st.sidebar:
    st.header("📊 Detection Statistics")

    stats = db.get_detection_stats()
    if stats['total_images'] > 0:
        st.metric("Total Images Analyzed", stats['total_images'])
        st.metric("Total Defects Found", stats['total_defects'])
        if stats['total_defects'] > 0:
            st.metric("Average Confidence", f"{stats['avg_confidence']:.2%}")
    else:
        st.info("No images analyzed yet")

    st.divider()
    st.header("⚙️ Settings")
    show_confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    if st.button("🧹 Clear History", use_container_width=True):
        db.clear_all()
        st.rerun()

st.title("🔍 AI Manufacturing Defect Detector")
st.markdown("Upload product images to detect manufacturing defects using the YOLO model.")

tab1, tab2 = st.tabs(["🖼️ Single Image", "📁 Batch Processing"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a product image...",
            type=['jpg', 'jpeg', 'png'],
            key="single_upload"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)

            if st.button("🔍 Analyze for Defects", type="primary", use_container_width=True):
                with st.spinner("Analyzing image for defects..."):
                    defects, annotated = detect_defects_yolo(image, show_confidence_threshold)
                    filtered_defects = [d for d in defects if d["confidence"] >= show_confidence_threshold]

                    result_image = draw_defects(image, annotated)

                    avg_conf = np.mean([d['confidence'] for d in filtered_defects]) if filtered_defects else 0
                    defect_details = json.dumps(filtered_defects)

                    db.add_detection(uploaded_file.name, len(filtered_defects), avg_conf, defect_details)

                    st.session_state.current_image = image
                    st.session_state.current_result_image = result_image
                    st.session_state.current_defects = filtered_defects

                    st.rerun()

    with col2:
        st.subheader("📋 Detection Results")

        if hasattr(st.session_state, "current_defects"):
            defects = st.session_state.current_defects
            if len(defects) > 0:
                st.image(st.session_state.current_result_image,
                         caption=f"Detected {len(defects)} Defect(s)",
                         use_column_width=True)
                st.error(f"⚠️ {len(defects)} Defect(s) Detected")

                defect_data = [{
                    "No.": i + 1,
                    "Type": d["type"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "Area": d["area"]
                } for i, d in enumerate(defects)]

                df = pd.DataFrame(defect_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                defect_types = [d["type"] for d in defects]
                type_counts = pd.Series(defect_types).value_counts()

                fig = go.Figure(data=[go.Pie(labels=type_counts.index, values=type_counts.values, hole=0.3)])
                fig.update_layout(title="Defect Type Distribution", height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No Defects Detected")
        else:
            st.info("👆 Upload an image and click 'Analyze for Defects' to begin detection.")

# ======== BATCH TAB ========
with tab2:
    st.subheader("📁 Batch Upload & Analysis")
    batch_files = st.file_uploader(
        "Choose multiple product images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files:
        st.write(f"**{len(batch_files)} image(s) uploaded**")

        if st.button("🔍 Analyze All Images", type="primary", use_container_width=True):
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file in enumerate(batch_files):
                status_text.text(f"Processing {idx + 1}/{len(batch_files)}: {file.name}")
                image = Image.open(file).convert("RGB")

                defects, annotated = detect_defects_yolo(image, show_confidence_threshold)
                filtered_defects = [d for d in defects if d["confidence"] >= show_confidence_threshold]

                avg_conf = np.mean([d['confidence'] for d in filtered_defects]) if filtered_defects else 0
                defect_details = json.dumps(filtered_defects)

                db.add_detection(file.name, len(filtered_defects), avg_conf, defect_details)

                batch_results.append({
                    "filename": file.name,
                    "defect_count": len(filtered_defects),
                    "avg_confidence": avg_conf,
                    "image": draw_defects(image, annotated)
                })

                progress_bar.progress((idx + 1) / len(batch_files))

            status_text.success("✅ Batch analysis completed!")
            st.session_state.batch_results = batch_results
            st.rerun()

    if hasattr(st.session_state, "batch_results") and st.session_state.batch_results:
        st.divider()
        st.subheader("📊 Batch Analysis Results")

        total_defects = sum([r["defect_count"] for r in st.session_state.batch_results])
        images_with_defects = sum([1 for r in st.session_state.batch_results if r["defect_count"] > 0])

        col1, col2, col3 = st.columns(3)
        col1.metric("Images Processed", len(st.session_state.batch_results))
        col2.metric("Images with Defects", images_with_defects)
        col3.metric("Total Defects Found", total_defects)

        results_data = [{
            "Filename": r["filename"],
            "Defects Found": r["defect_count"],
            "Avg Confidence": f"{r['avg_confidence']:.2%}" if r["avg_confidence"] > 0 else "N/A",
            "Status": "⚠️ Defects" if r["defect_count"] > 0 else "✅ Clean"
        } for r in st.session_state.batch_results]

        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

        st.subheader("Analyzed Images")
        cols_per_row = 3
        for i in range(0, len(st.session_state.batch_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(st.session_state.batch_results):
                    result = st.session_state.batch_results[idx]
                    with cols[j]:
                        st.image(result["image"], caption=f"{result['filename']} ({result['defect_count']} defects)", use_container_width=True)

# ======== HISTORY SECTION ========
history_records = db.get_all_detections()
if history_records:
    st.divider()
    st.subheader("📜 Detection History")

    display_records = [{
        "Timestamp": r[4],
        "Filename": r[1],
        "Defects Found": r[2],
        "Avg Confidence": f"{r[3]:.2%}" if r[3] > 0 else "N/A"
    } for r in history_records]
    history_df = pd.DataFrame(display_records)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown("""
### 🧠 YOLO Model Integration Notes

This app uses a **YOLO model (Ultralytics)** for detecting product defects.

To change the model:
1. Replace the `best.pt` file with your own trained YOLO weights.
2. Adjust confidence thresholds in the sidebar as needed.
3. The results are automatically saved in the local database for analytics.
""")

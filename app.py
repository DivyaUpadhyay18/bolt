"""
Streamlit dashboard for automated bolt hole detection.
Live video processing with persistent hole numbering and hole navigator.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from detector import BoltHoleDetector
from tracker import BoltHoleTracker
from panel_finder import find_bscan_roi, draw_roi_overlay
from utils import plot_projection_debug, draw_masks_overlay, draw_numbered_holes


st.set_page_config(page_title="Bolt Hole Detection", layout="wide")
st.title("🔍 Automated Bolt Hole Detection System")
st.markdown("Real-time video analysis with persistent hole numbering")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📤 Video Upload")
    uploaded_file = st.file_uploader(
        "Upload B-Scan Video",
        type=["mp4", "avi", "mov"],
        help="SRT BScan video file"
    )
    
    st.markdown("---")
    st.subheader("⚙️ Detection Parameters")
    
    frame_skip = st.slider(
        "Process every N frames",
        1, 10, 2,
        help="Skip frames for speed"
    )
    
    sigma = st.slider(
        "Smoothing sigma",
        3, 25, 10,
        step=1,
        help="Higher = detects more scattered dots"
    )
    
    min_dist = st.slider(
        "Min pixels between holes",
        10, 60, 18,
        step=1,
        help="Minimum spacing"
    )
    
    prominence = st.slider(
        "Peak prominence",
        0.05, 2.0, 0.2,
        step=0.05,
        help="Detection sensitivity"
    )
    
    st.markdown("---")
    st.subheader("📊 Display Options")
    show_debug = st.checkbox("Show projection debug graph", value=False)
    show_roi = st.checkbox("Show detection zone outline", value=True)

# ============================================================================
# MAIN PROCESSING
# ============================================================================

if uploaded_file is not None:
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_video_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open video file")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            st.subheader("📊 Video Info")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames", total_frames)
            col2.metric("FPS", f"{fps:.1f}")
            col3.metric("Resolution", f"{width}×{height}")
            col4.metric("Duration", f"{total_frames/fps:.1f}s" if fps > 0 else "?")
            
            if st.button("🚀 Start Processing", key="process_btn"):
                
                # Initialize
                detector = BoltHoleDetector(
                    sigma=sigma,
                    min_distance=min_dist,
                    prominence=prominence
                )
                tracker = BoltHoleTracker(max_distance=35, max_missing_frames=6)
                
                stored_frames = {}  # {frame_num: annotated_bgr_frame}
                frame_results = []
                
                # Placeholders
                frame_col, count_col = st.columns([3, 1])
                frame_display = frame_col.empty()
                count_display = count_col.empty()
                progress_bar = st.progress(0)
                chart_ph = st.empty()
                debug_ph = st.empty()
                
                # Process frames
                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_num += 1
                    if frame_num % frame_skip != 0:
                        continue
                    
                    # 1. Find ROI
                    roi_dict = find_bscan_roi(frame)
                    roi = roi_dict["roi"]
                    
                    # 2. Detect holes
                    result = detector.detect(roi)
                    
                    # 3. Track holes
                    active = tracker.update(result["bolt_hole_positions"], frame_num)
                    
                    # 4. Draw numbered labels on ROI
                    ann_roi = draw_numbered_holes(result["annotated_roi"], active)
                    
                    # 5. Paste ROI back into full frame
                    ann_frame = frame.copy()
                    ann_frame[roi_dict["y_top"]:roi_dict["y_bot"],
                              roi_dict["x_left"]:roi_dict["x_right"]] = ann_roi
                    
                    # 6. Draw ROI outline if requested
                    if show_roi:
                        ann_frame = draw_roi_overlay(ann_frame, roi_dict)
                    
                    # 7. Draw summary banner
                    total_unique = tracker.next_id - 1
                    banner = f"ACTIVE: {len(active)}  |  TOTAL UNIQUE: {total_unique}"
                    
                    # Black outline
                    cv2.putText(ann_frame, banner, (30, 35),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    # White text
                    cv2.putText(ann_frame, banner, (30, 35),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # 8. Store frame (every 5th)
                    if len(frame_results) % 5 == 0:
                        stored_frames[frame_num] = ann_frame.copy()
                    
                    # 9. Collect results
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frame_results.append({
                        "frame": frame_num,
                        "time_sec": round(ts, 2),
                        "active_holes": len(active),
                        "total_unique": total_unique,
                    })
                    
                    # 10. Update displays
                    frame_display.image(
                        cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                        caption=f"Frame {frame_num}"
                    )
                    
                    count_display.markdown(
                        f"<div style='text-align: center;'>"
                        f"<h2>🟢 {len(active)}</h2>"
                        f"<p>Active holes</p>"
                        f"<hr>"
                        f"<p><b>Total unique:</b> {total_unique}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    progress_bar.progress(min(frame_num / total_frames, 1.0))
                    
                    # Update chart every 15 frames
                    if len(frame_results) % 15 == 0:
                        df_chart = pd.DataFrame(frame_results)
                        chart_ph.line_chart(df_chart.set_index("frame")["active_holes"])
                    
                    # Show debug graph if enabled
                    if show_debug and "combined_signal" in result:
                        debug_result = {
                            "purple_proj": result["purple_proj"],
                            "red_proj": result["red_proj"],
                            "grey_proj": result["grey_proj"],
                            "combined_signal": result["combined_signal"],
                            "bolt_hole_positions": result["bolt_hole_positions"],
                        }
                        debug_img = plot_projection_debug(debug_result)
                        debug_ph.image(
                            cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                            caption="Projection signals (red dashed = detected holes)"
                        )
                
                cap.release()
                
                # ================================================================
                # RESULTS SUMMARY
                # ================================================================
                
                st.success(f"✅ Processing complete — {len(frame_results)} frames analysed")
                
                st.subheader("📈 Results")
                if frame_results:
                    df = pd.DataFrame(frame_results)
                    st.dataframe(df, use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frames Processed", len(frame_results))
                    col2.metric("Max Active", int(df["active_holes"].max()))
                    col3.metric("Total Unique", int(df["total_unique"].iloc[-1]))
                    col4.metric("Avg Active", f"{df['active_holes'].mean():.1f}")
                    
                    st.download_button(
                        "📥 Download Results CSV",
                        df.to_csv(index=False),
                        "results.csv",
                        "text/csv"
                    )
                    
                    st.subheader("📊 Active Holes Over Time")
                    st.line_chart(df.set_index("frame")["active_holes"])
                
                # ================================================================
                # HOLE NAVIGATOR
                # ================================================================
                
                st.markdown("---")
                st.header("🔍 Individual Hole Navigator")
                
                if tracker.hole_history:
                    
                    # Summary table
                    history_data = []
                    for h in tracker.hole_history:
                        history_data.append({
                            "Hole Label": h["label"],
                            "First Frame": h["first_frame"],
                            "Last Frame": h["last_frame"],
                            "Duration (frames)": h["last_frame"] - h["first_frame"],
                            "Position X": h["max_cx"],
                            "Position Y": h["max_cy"],
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    st.subheader(f"Total unique bolt holes: {len(tracker.hole_history)}")
                    st.dataframe(history_df, use_container_width=True)
                    
                    # Hole selector
                    st.subheader("Jump to a specific hole")
                    hole_labels = [h["label"] for h in tracker.hole_history]
                    selected_label = st.selectbox(
                        "Select hole to inspect:",
                        hole_labels,
                        key="hole_select"
                    )
                    
                    # Find selected hole
                    selected = next(h for h in tracker.hole_history if h["label"] == selected_label)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Hole ID", selected["label"])
                    col2.metric("First Frame", selected["first_frame"])
                    col3.metric("Last Frame", selected["last_frame"])
                    col4.metric("Active for", selected["last_frame"] - selected["first_frame"])
                    
                    # Show frame where hole first appeared
                    if selected["first_frame"] in stored_frames:
                        st.markdown(f"**Frame where {selected_label} first appeared:**")
                        frame_rgb = cv2.cvtColor(
                            stored_frames[selected["first_frame"]], cv2.COLOR_BGR2RGB
                        )
                        st.image(frame_rgb, use_column_width=True,
                                caption=f"{selected_label} — first appearance")
                    else:
                        # Find nearest stored frame
                        nearest = min(stored_frames.keys(),
                                     key=lambda k: abs(k - selected["first_frame"]))
                        frame_rgb = cv2.cvtColor(
                            stored_frames[nearest], cv2.COLOR_BGR2RGB
                        )
                        st.image(frame_rgb, use_column_width=True,
                                caption=f"{selected_label} — nearest frame {nearest}")
                    
                    # Download hole history
                    st.download_button(
                        "📥 Download Hole History CSV",
                        history_df.to_csv(index=False),
                        "hole_history.csv",
                        "text/csv"
                    )
                
                else:
                    st.info("No holes detected in this video.")
    
    finally:
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

else:
    st.info("👆 Upload a B-Scan video to get started")

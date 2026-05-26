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
import time
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
    
    # Frame processing options
    frame_skip = st.selectbox(
        "Process every N frames",
        options=[1, 2, 3, 4, 5, 10, 15, 20, 30],
        index=1,  # Default to 2
        help="Skip frames for speed — higher = faster but might miss some holes"
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
        "Peak prominence — lower catches weaker holes",
        0.05, 1.0, 0.2,
        step=0.05,
        help="Detection sensitivity"
    )
    
    st.markdown("---")
    st.subheader("⚡ Speed Optimization")
    
    roi_cache_interval = st.select_slider(
        "ROI re-detection frequency",
        options=[1, 5, 10, 15, 20, 30, 50, 100],
        value=30,
        help="Re-detect ROI every N frames (higher = faster, lower = more accurate)"
    )
    
    resolution_scale = st.selectbox(
        "Process resolution scale",
        options=["Full (100%)", "75%", "50%", "33%"],
        index=0,
        help="Process at lower resolution for speed boost (trades accuracy for speed)"
    )
    
    resolution_scale_map = {
        "Full (100%)": 1.0,
        "75%": 0.75,
        "50%": 0.5,
        "33%": 0.33
    }
    scale_factor = resolution_scale_map[resolution_scale]
    
    display_update_freq = st.select_slider(
        "Display refresh rate",
        options=[1, 5, 10, 15, 30],
        value=1,
        help="Update screen every N processed frames (higher = faster rendering)"
    )
    
    st.markdown("---")
    st.subheader("📊 Display Options")
    show_debug = st.checkbox("Show projection debug graph", value=False)
    show_roi = st.checkbox("Show detection zone outline", value=True)
    headless_mode = st.checkbox("Headless mode (no display, faster processing)", value=False)

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
                
                cached_roi_dict = None  # Cache ROI boundaries
                
                stored_frames = {}  # {frame_num: annotated_bgr_frame}
                frame_results = []
                
                # Placeholders
                if not headless_mode:
                    frame_col, count_col = st.columns([3, 1])
                    frame_display = frame_col.empty()
                    count_display = count_col.empty()
                progress_bar = st.progress(0)
                if not headless_mode:
                    chart_ph = st.empty()
                    debug_ph = st.empty()
                
                # Performance timing
                frame_times = []
                start_time = time.time()
                
                # Process frames
                frame_num = 0
                processed_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_num += 1
                    if frame_num % frame_skip != 0:
                        continue
                    
                    frame_start = time.time()
                    processed_count += 1
                    
                    # Scale frame for faster processing
                    if scale_factor < 1.0:
                        frame_scaled = cv2.resize(
                            frame,
                            (int(frame.shape[1] * scale_factor),
                             int(frame.shape[0] * scale_factor)),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        frame_scaled = frame
                    
                    # ================================================================
                    # 1. FIND ROI — with caching at new interval
                    # ================================================================
                    if (cached_roi_dict is None or
                            not cached_roi_dict["found"] or
                            processed_count % roi_cache_interval == 0):
                        cached_roi_dict = find_bscan_roi(frame_scaled)
                    
                    # Skip frame if ROI detection failed
                    if not cached_roi_dict["found"]:
                        print(f"[FRAME {frame_num}] ROI not found, skipping")
                        continue
                    
                    # Re-crop ROI from current frame using cached boundaries
                    roi_dict = cached_roi_dict.copy()
                    roi_dict["roi"] = frame_scaled[
                        cached_roi_dict["y_top"]:cached_roi_dict["y_bot"],
                        cached_roi_dict["x_left"]:cached_roi_dict["x_right"]
                    ]
                    roi = roi_dict["roi"]
                    
                    # ================================================================
                    # 2. DETECT HOLES
                    # ================================================================
                    result = detector.detect(roi, sigma=sigma, min_distance=min_dist, 
                                             prominence=prominence)
                    
                    # ================================================================
                    # 3. TRACK HOLES
                    # ================================================================
                    active = tracker.update(result["bolt_hole_positions"], frame_num)
                    
                    # ================================================================
                    # 4. DRAW NUMBERED LABELS (skip in headless mode)
                    # ================================================================
                    if not headless_mode:
                        ann_roi = draw_numbered_holes(result["annotated_roi"], active)
                    else:
                        ann_roi = result["annotated_roi"]
                    
                    # ================================================================
                    # 5. PASTE ROI BACK INTO FULL FRAME (skip in headless mode)
                    # ================================================================
                    if not headless_mode:
                        ann_frame = frame_scaled.copy()
                        ann_frame[roi_dict["y_top"]:roi_dict["y_bot"],
                                  roi_dict["x_left"]:roi_dict["x_right"]] = ann_roi
                        
                        # ================================================================
                        # 6. DRAW ROI OUTLINE
                        # ================================================================
                        if show_roi:
                            ann_frame = draw_roi_overlay(ann_frame, roi_dict)
                        
                        # ================================================================
                        # 7. DRAW SUMMARY BANNER
                        # ================================================================
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
                    else:
                        total_unique = tracker.next_id - 1
                    
                    # ================================================================
                    # 8. STORE FRAME FOR NAVIGATOR
                    # ================================================================
                    if len(frame_results) % 5 == 0 and not headless_mode:
                        stored_frames[frame_num] = ann_frame.copy()
                    
                    # ================================================================
                    # 9. COLLECT RESULTS
                    # ================================================================
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frame_results.append({
                        "frame": frame_num,
                        "time_sec": round(ts, 2),
                        "active_holes": len(active),
                        "total_unique": total_unique,
                    })
                    
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                    
                    # ================================================================
                    # 10. UPDATE DISPLAYS (only if display_update_freq reached)
                    # ================================================================
                    if not headless_mode and len(frame_results) % display_update_freq == 0:
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
                            f"<hr>"
                            f"<p style='font-size: 0.8em;'>Frame time: {frame_time*1000:.1f}ms</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # ================================================================
                        # 11. UPDATE CHART (every 15 frames)
                        # ================================================================
                        if len(frame_results) % 15 == 0:
                            df_chart = pd.DataFrame(frame_results)
                            chart_ph.line_chart(df_chart.set_index("frame")["active_holes"])
                        
                        # ================================================================
                        # 12. SHOW DEBUG GRAPH (if enabled)
                        # ================================================================
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
                    
                    progress_bar.progress(min(frame_num / total_frames, 1.0))
                
                cap.release()
                
                # ================================================================
                # PERFORMANCE METRICS
                # ================================================================
                elapsed_time = time.time() - start_time
                avg_frame_time = np.mean(frame_times) if frame_times else 0
                fps_achieved = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                if headless_mode:
                    st.success(f"✅ Headless processing complete!")
                    st.info(f"⏱️ **Performance**: {processed_count} frames in {elapsed_time:.1f}s "
                           f"({fps_achieved:.1f} FPS avg, {avg_frame_time*1000:.1f}ms/frame)")
                else:
                    st.success(f"✅ Processing complete — {len(frame_results)} frames analysed")
                
                # ================================================================
                # RESULTS SUMMARY
                # ================================================================
                
                st.subheader("📈 Results")
                if frame_results:
                    df = pd.DataFrame(frame_results)
                    st.dataframe(df, use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frames Processed", len(frame_results))
                    col2.metric("Max Active", int(df["active_holes"].max()))
                    col3.metric("Total Unique", int(df["total_unique"].iloc[-1]))
                    col4.metric("Avg Active", f"{df['active_holes'].mean():.1f}")
                    
                    st.markdown("---")
                    st.subheader("⚡ Performance Metrics")
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                    perf_col1.metric("Total Time", f"{elapsed_time:.1f}s")
                    perf_col2.metric("Avg Frame Time", f"{avg_frame_time*1000:.1f}ms")
                    perf_col3.metric("Achieved FPS", f"{fps_achieved:.1f}")
                    perf_col4.metric("Skip Ratio", f"1:{frame_skip}")
                    
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
                
                if headless_mode:
                    st.info("ℹ️ Hole navigator not available in headless mode")
                elif tracker.hole_history:
                    
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
        # Clean up temp file with error handling
        if os.path.exists(temp_video_path):
            try:
                cap.release()  # Ensure video is released
                time.sleep(0.5)  # Small delay to release file lock
                os.unlink(temp_video_path)
            except Exception as e:
                print(f"[WARNING] Could not delete temp file: {e}")

else:
    st.info("👆 Upload a B-Scan video to get started")

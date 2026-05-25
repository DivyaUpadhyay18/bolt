"""
Demo script: Extract frames from video and run bolt hole detection.
Shows integration with video processing pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from detector import BoltHoleDetector
from utils import plot_projection_debug


VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"


def extract_roi_from_frame(frame_bgr, top_blue_y=None, bottom_blue_y=None, red_y=None):
    """
    Extract ROI from a full frame between the blue and red reference lines.
    
    If line positions not provided, automatically detect them.
    
    Args:
        frame_bgr: Full frame (BGR)
        top_blue_y: Y coordinate of top blue line (auto-detect if None)
        bottom_blue_y: Y coordinate of bottom blue line (auto-detect if None)
        red_y: Y coordinate of red line (auto-detect if None)
    
    Returns:
        roi_bgr: Extracted ROI region
        y_start: Y coordinate of ROI top
        y_end: Y coordinate of ROI bottom
    """
    
    # Auto-detect reference lines if not provided
    if top_blue_y is None or bottom_blue_y is None or red_y is None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Detect blue lines
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        blue_rows = np.where(blue_mask.sum(axis=1) > 100)[0]
        
        if len(blue_rows) > 0:
            top_blue_y = blue_rows[0]
            bottom_blue_y = blue_rows[-1]
        else:
            top_blue_y = 50
            bottom_blue_y = 600
        
        # Detect red line
        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                  cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_rows = np.where(red_mask.sum(axis=1) > 100)[0]
        
        if len(red_rows) > 0:
            red_y = red_rows[-1]
        else:
            red_y = 950
    
    # Extract ROI (between bottom blue and red line)
    y_start = bottom_blue_y + 5
    y_end = red_y - 5
    
    roi_bgr = frame_bgr[y_start:y_end, :].copy()
    
    return roi_bgr, y_start, y_end


def run_demo_on_video_frames(video_path, frame_numbers=[100, 500, 1000, 2000, 3000]):
    """
    Extract and detect bolt holes in specific video frames.
    
    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices to process
    """
    
    print("="*70)
    print("BOLT HOLE DETECTOR — Video Demo")
    print("="*70)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {Path(video_path).name}")
    print(f"  Resolution: {width}×{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create detector
    detector = BoltHoleDetector(sigma=8, min_distance=20, prominence=0.3)
    print(f"\nDetector parameters:")
    print(f"  σ=8, min_distance=20, prominence=0.3")
    
    # Process frames
    print(f"\nProcessing {len(frame_numbers)} frames...\n")
    
    all_results = []
    
    for frame_num in frame_numbers:
        if frame_num >= total_frames:
            print(f"Frame {frame_num}: SKIP (exceeds total)")
            continue
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame_bgr = cap.read()
        
        if not ret:
            print(f"Frame {frame_num}: ERROR reading frame")
            continue
        
        # Extract ROI
        roi_bgr, y_start, y_end = extract_roi_from_frame(frame_bgr)
        
        # Detect
        result = detector.detect(roi_bgr)
        
        # Store result
        all_results.append({
            'frame': frame_num,
            'count': result['bolt_hole_count'],
            'positions': result['bolt_hole_positions'],
            'roi': roi_bgr,
            'result': result
        })
        
        # Print result
        time_sec = frame_num / fps
        print(f"Frame {frame_num:4d} ({time_sec:7.2f}s): {result['bolt_hole_count']} holes detected")
        
        # Save annotated ROI
        frame_output = output_dir / f"frame_{frame_num:05d}"
        frame_output.mkdir(exist_ok=True)
        
        annotated_path = frame_output / "annotated.png"
        cv2.imwrite(str(annotated_path), result['annotated_roi'])
        
        debug_path = frame_output / "projection.png"
        cv2.imwrite(str(debug_path), plot_projection_debug(result))
        
        print(f"    Saved to: {frame_output}")
        print()
    
    cap.release()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Frame':<8} {'Time (s)':<12} {'Holes':<8} {'Positions':<30}")
    print("-"*70)
    
    for r in all_results:
        time_sec = r['frame'] / fps
        positions_str = ", ".join([f"({x},{y})" for x, y in r['positions'][:3]])
        if len(r['positions']) > 3:
            positions_str += f", +{len(r['positions'])-3} more"
        
        print(f"{r['frame']:<8} {time_sec:<12.2f} {r['count']:<8} {positions_str:<30}")
    
    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Optional: allow custom video path and frame numbers
    video = VIDEO_PATH
    frames = [100, 500, 1000, 2000, 3000]
    
    if len(sys.argv) > 1:
        video = sys.argv[1]
    if len(sys.argv) > 2:
        frames = [int(x) for x in sys.argv[2].split(',')]
    
    run_demo_on_video_frames(video, frames)

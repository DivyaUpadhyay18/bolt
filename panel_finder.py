"""
Find the B-Scan detection zone in video frames.
Visual detection based on blue/red lines and grey panel background.
Works with any zoom level or frame position.
"""

import cv2
import numpy as np


def find_bscan_roi(frame):
    """
    Extract ONLY the B-Scan scanning region from frame.
    
    The B-Scan is the DATA SWEEP AREA bounded by:
    - Top: SECOND blue horizontal line
    - Bottom: RED horizontal line
    - Left: Start of the grey scanning grid
    - Right: End of the grey scanning grid (NOT extending to data table)
    
    Works at ANY zoom level by detecting visual boundaries only.
    
    Returns:
        dict with keys: roi, y_top, y_bot, x_left, x_right, found (bool)
    """
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("[ROI] ========== SEARCHING FOR B-SCAN REGION ==========")
    
    # ========================================================================
    # STEP 1 — Find horizontal BLUE lines (top boundary)
    # Look for SOLID horizontal blue lines, not scattered pixels
    # ========================================================================
    print("[ROI] Step 1: Detecting BLUE horizontal lines...")
    blue_lines = _find_horizontal_lines(frame_rgb, color_type='blue', min_strength=40)
    
    if len(blue_lines) < 2:
        print(f"[ROI] ERROR: Found only {len(blue_lines)} blue lines, need 2")
        return _get_adaptive_fallback_roi(frame)
    
    blue_lines = sorted(blue_lines)
    y_second_blue = blue_lines[1]
    print(f"[ROI] ✓ Blue lines found: {blue_lines}")
    print(f"[ROI] ✓ Using SECOND blue line at y={y_second_blue}")
    
    # ========================================================================
    # STEP 2 — Find horizontal RED line (bottom boundary)
    # ========================================================================
    print("[ROI] Step 2: Detecting RED horizontal line...")
    red_lines = _find_horizontal_lines(frame_rgb, color_type='red', min_strength=40)
    
    if len(red_lines) == 0:
        print("[ROI] ERROR: No red line found")
        return _get_adaptive_fallback_roi(frame)
    
    red_lines = sorted(red_lines)
    
    # Find first red line BELOW second blue line
    y_red = None
    for ry in red_lines:
        if ry > y_second_blue + 10:
            y_red = ry
            break
    
    if y_red is None:
        print(f"[ROI] ERROR: Red line not below blue line. Red: {red_lines}, Blue-2: {y_second_blue}")
        return _get_adaptive_fallback_roi(frame)
    
    print(f"[ROI] ✓ Red line found at y={y_red}")
    
    # ========================================================================
    # STEP 3 — Set Y boundaries with minimal margins
    # ========================================================================
    y_top = y_second_blue + 1
    y_bot = y_red - 1
    
    print(f"[ROI] ✓ Y boundaries: [{y_top}, {y_bot}] (height={y_bot-y_top})")
    
    if (y_bot - y_top) < 5:
        print(f"[ROI] ERROR: Detection zone too small")
        return _get_adaptive_fallback_roi(frame)
    
    # ========================================================================
    # STEP 4 — Find X boundaries by detecting the GREY PANEL
    # The B-Scan panel has consistent grey/dark background
    # Extend only as far as grey extends, not into white table
    # ========================================================================
    print("[ROI] Step 3: Detecting HORIZONTAL boundaries of grey B-Scan panel...")
    x_left, x_right = _find_grey_panel_boundaries(frame_rgb, y_top, y_bot)
    
    print(f"[ROI] ✓ X boundaries: [{x_left}, {x_right}] (width={x_right-x_left})")
    
    if (x_right - x_left) < 20:
        print(f"[ROI] ERROR: Detection zone too narrow")
        return _get_adaptive_fallback_roi(frame)
    
    # ========================================================================
    # STEP 5 — FINAL CHECK: Verify we have actual B-Scan content
    # ========================================================================
    roi = frame[y_top:y_bot, x_left:x_right]
    roi_rgb = frame_rgb[y_top:y_bot, x_left:x_right]
    
    # Check if ROI has any non-white content (should have B-Scan data)
    is_white = (roi_rgb[:,:,0] > 240) & (roi_rgb[:,:,1] > 240) & (roi_rgb[:,:,2] > 240)
    white_percentage = np.sum(is_white) / (roi_rgb.shape[0] * roi_rgb.shape[1])
    
    if white_percentage > 0.8:
        print(f"[ROI] ERROR: ROI is {white_percentage*100:.1f}% white - not a B-Scan region")
        return _get_adaptive_fallback_roi(frame)
    
    print(f"[ROI] ✓ ROI content check: {100-white_percentage*100:.1f}% is non-white")
    
    print(f"[ROI] ========== B-SCAN REGION DETECTED ==========")
    print(f"[ROI] auto=YES y_top={y_top} y_bot={y_bot} x_left={x_left} x_right={x_right} "
          f"shape={roi.shape}")
    
    return {
        "roi": roi,
        "y_top": y_top,
        "y_bot": y_bot,
        "x_left": x_left,
        "x_right": x_right,
        "found": True,
    }


def _find_horizontal_lines(frame_rgb, color_type='blue', min_strength=40):
    """
    Find SOLID horizontal lines of specific color.
    
    Looks for rows where the entire row (or majority) has strong color dominance.
    This finds ACTUAL lines, not scattered pixels.
    
    Args:
        frame_rgb: RGB frame
        color_type: 'blue' or 'red'
        min_strength: minimum color dominance to count as a line
    
    Returns:
        list of y-coordinates of detected lines
    """
    H, W = frame_rgb.shape[:2]
    
    line_scores = []
    
    for y in range(H):
        row = frame_rgb[y, :, :]
        
        r = row[:, 0].astype(float)
        g = row[:, 1].astype(float)
        b = row[:, 2].astype(float)
        
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        
        if color_type == 'blue':
            # Blue line: B is much higher than R and G
            dominance = avg_b - max(avg_r, avg_g)
        elif color_type == 'red':
            # Red line: R is much higher than G and B
            dominance = avg_r - max(avg_g, avg_b)
        else:
            dominance = 0
        
        if dominance >= min_strength:
            line_scores.append((y, dominance))
    
    # Cluster consecutive high-scoring rows
    lines = _cluster_lines(line_scores, distance_threshold=4)
    
    return lines


def _find_grey_panel_boundaries(frame_rgb, y_top, y_bot):
    """
    Find where the grey B-Scan panel starts and ends horizontally.
    
    Grey panel: R≈G≈B, value between 80-220 (not pure white, not pure black)
    Table area: pure white (R,G,B > 240) or text/cells
    
    Scan from left and right to find exact boundaries.
    
    Args:
        frame_rgb: RGB frame
        y_top: Top of B-Scan zone
        y_bot: Bottom of B-Scan zone
    
    Returns:
        (x_left, x_right) tuple - boundaries of grey panel
    """
    H, W = frame_rgb.shape[:2]
    
    # Sample multiple rows from the zone for robustness
    sample_rows = []
    for i in range(5):
        y = y_top + int((y_bot - y_top) * i / 5)
        if y < y_bot:
            sample_rows.append(y)
    
    if not sample_rows:
        sample_rows = [(y_top + y_bot) // 2]
    
    # ========================================================================
    # Find LEFT boundary: scan left-to-right for first grey pixel
    # ========================================================================
    x_left = 0
    for x in range(W):
        grey_count = 0
        
        for y in sample_rows:
            r, g, b = int(frame_rgb[y, x, 0]), int(frame_rgb[y, x, 1]), int(frame_rgb[y, x, 2])
            
            # Check if this is grey (B-Scan panel color)
            is_grey = (
                (abs(int(r) - int(g)) < 20) and
                (abs(int(g) - int(b)) < 20) and
                (abs(int(r) - int(b)) < 20) and
                (80 <= r <= 220)
            )
            
            if is_grey:
                grey_count += 1
        
        # If MOST rows show grey here, we found the left edge
        if grey_count >= len(sample_rows) * 0.6:
            x_left = x
            print(f"[ROI] Left boundary: x={x_left} (grey in {grey_count}/{len(sample_rows)} rows)")
            break
    
    # ========================================================================
    # Find RIGHT boundary: scan right-to-left for last grey pixel
    # ========================================================================
    x_right = W
    for x in range(W - 1, 0, -1):
        grey_count = 0
        
        for y in sample_rows:
            r, g, b = int(frame_rgb[y, x, 0]), int(frame_rgb[y, x, 1]), int(frame_rgb[y, x, 2])
            
            # Check if grey
            is_grey = (
                (abs(int(r) - int(g)) < 20) and
                (abs(int(g) - int(b)) < 20) and
                (abs(int(r) - int(b)) < 20) and
                (80 <= r <= 220)
            )
            
            if is_grey:
                grey_count += 1
        
        # If MOST rows show grey here, we found the right edge
        if grey_count >= len(sample_rows) * 0.6:
            x_right = x + 1
            print(f"[ROI] Right boundary: x={x_right} (grey in {grey_count}/{len(sample_rows)} rows)")
            break
    
    # Sanity check
    if x_right <= x_left:
        print(f"[ROI] ERROR: Invalid boundaries x_left={x_left} x_right={x_right}")
        # Use broad fallback
        x_left = int(W * 0.05)
        x_right = int(W * 0.35)
    
    # Hard limit: never exceed 40% of frame (B-Scan should be small)
    x_right = min(x_right, int(W * 0.40))
    
    return x_left, x_right



def _cluster_lines(line_scores, distance_threshold=4):
    """
    Cluster consecutive scored lines that are close together.
    
    Args:
        line_scores: list of (y, score) tuples
        distance_threshold: rows within this distance = same line
    
    Returns:
        list of clustered y-coordinates
    """
    if not line_scores:
        return []
    
    line_scores = sorted(line_scores, key=lambda x: x[0])
    
    clusters = []
    current_cluster = [line_scores[0][0]]
    
    for i in range(1, len(line_scores)):
        y, score = line_scores[i]
        
        if y - current_cluster[-1] <= distance_threshold:
            current_cluster.append(y)
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [y]
    
    if current_cluster:
        clusters.append(int(np.mean(current_cluster)))
    
    return sorted(set(clusters))


def _find_horizontal_lines(frame_rgb, color_type='blue', min_strength=40):
    """
    Find SOLID horizontal lines of specific color.
    
    Looks for rows where the entire row (or majority) has strong color dominance.
    This finds ACTUAL lines, not scattered pixels.
    
    Args:
        frame_rgb: RGB frame
        color_type: 'blue' or 'red'
        min_strength: minimum color dominance to count as a line
    
    Returns:
        list of y-coordinates of detected lines
    """
    H, W = frame_rgb.shape[:2]
    
    line_scores = []
    
    for y in range(H):
        row = frame_rgb[y, :, :]
        
        r = row[:, 0].astype(float)
        g = row[:, 1].astype(float)
        b = row[:, 2].astype(float)
        
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        
        if color_type == 'blue':
            # Blue line: B is much higher than R and G
            dominance = avg_b - max(avg_r, avg_g)
        elif color_type == 'red':
            # Red line: R is much higher than G and B
            dominance = avg_r - max(avg_g, avg_b)
        else:
            dominance = 0
        
        if dominance >= min_strength:
            line_scores.append((y, dominance))
    
    # Cluster consecutive high-scoring rows
    lines = _cluster_lines(line_scores, distance_threshold=4)
    
    return lines


def _get_fallback_roi(frame):
    """
    Return a fallback ROI when line detection fails.
    Uses a broad central region as a safe default.
    
    Args:
        frame: BGR frame
    
    Returns:
        ROI dict with fallback coordinates
    """
    H, W = frame.shape[:2]
    
    print("[ROI] auto=FALLBACK - using broad central region")
    
    x_left = int(W * 0.05)   # 5% from left
    x_right = int(W * 0.50)  # 50% of width (safe limit)
    y_top = int(H * 0.20)    # 20% from top
    y_bot = int(H * 0.80)    # 80% from top (60% height)
    
    roi = frame[y_top:y_bot, x_left:x_right]
    
    print(f"[ROI] Fallback: y_top={y_top} y_bot={y_bot} x_left={x_left} x_right={x_right}")
    
    return {
        "roi": roi,
        "y_top": y_top,
        "y_bot": y_bot,
        "x_left": x_left,
        "x_right": x_right,
        "found": False,
    }


def _get_adaptive_fallback_roi(frame):
    """
    Adaptive fallback: tries to find ANY content in the frame
    and use a region around it.
    
    Args:
        frame: BGR frame
    
    Returns:
        ROI dict with adaptive fallback coordinates
    """
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("[ROI] auto=ADAPTIVE_FALLBACK - searching for content...")
    
    # Look for non-white/non-black regions (likely B-Scan area)
    r = frame_rgb[:, :, 0].astype(int)
    g = frame_rgb[:, :, 1].astype(int)
    b = frame_rgb[:, :, 2].astype(int)
    
    # Find pixels that are "content" (not pure white, not pure black)
    is_content = (
        ~((r > 240) & (g > 240) & (b > 240)) &  # Not white
        ~((r < 15) & (g < 15) & (b < 15))       # Not black
    )
    
    content_rows = np.where(is_content.any(axis=1))[0]
    content_cols = np.where(is_content.any(axis=0))[0]
    
    if len(content_rows) > 0 and len(content_cols) > 0:
        y_top = max(0, content_rows[0])
        y_bot = min(H, content_rows[-1])
        x_left = max(0, content_cols[0])
        x_right = min(int(W * 0.50), content_cols[-1])
        
        print(f"[ROI] Adaptive: found content y=[{y_top}, {y_bot}] x=[{x_left}, {x_right}]")
    else:
        # Fallback to broad region
        x_left = int(W * 0.05)
        x_right = int(W * 0.50)
        y_top = int(H * 0.20)
        y_bot = int(H * 0.80)
        print("[ROI] Adaptive fallback to broad region")
    
    roi = frame[y_top:y_bot, x_left:x_right]
    
    return {
        "roi": roi,
        "y_top": y_top,
        "y_bot": y_bot,
        "x_left": x_left,
        "x_right": x_right,
        "found": False,
    }


def draw_roi_overlay(frame, roi_dict):
    """
    Draw bright GREEN rectangle around detection zone on full frame.
    
    Args:
        frame: Full BGR frame
        roi_dict: Dict from find_bscan_roi()
    
    Returns:
        Annotated frame copy
    """
    ann_frame = frame.copy()
    
    x_left = roi_dict["x_left"]
    x_right = roi_dict["x_right"]
    y_top = roi_dict["y_top"]
    y_bot = roi_dict["y_bot"]
    
    # Draw green rectangle
    cv2.rectangle(ann_frame, (x_left, y_top), (x_right, y_bot),
                  (0, 255, 0), 3)
    
    # Draw label "DETECTION ZONE"
    label = "DETECTION ZONE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    
    text_x = x_left
    text_y = y_top - 5
    
    # Black background for text
    cv2.rectangle(ann_frame, (text_x - 2, text_y - th - 2),
                  (text_x + tw + 2, text_y + 2),
                  (0, 0, 0), -1)
    
    # Green text
    cv2.putText(ann_frame, label, (text_x, text_y),
                font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    return ann_frame

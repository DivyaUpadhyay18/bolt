import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FRAME-LOCAL SEQUENTIAL DETECTOR")
print(f"Each frame: find P→R, P→G, R→G transitions locally")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

def get_color_clusters(frame):
    """Get clusters for each color"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    color_clusters = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clusters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 60 or area > 25000:
                continue
            
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.3:
                continue
            
            clusters.append({'centroid': (cx, cy), 'area': area})
        
        color_clusters[color_name] = clusters
    
    return color_clusters

hole_count = 0
proximity_threshold = 60

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (counted {hole_count} holes so far)")
    
    color_clusters = get_color_clusters(frame)
    purple = color_clusters['purple']
    red = color_clusters['red']
    gray = color_clusters['gray']
    
    used_red = set()
    used_gray = set()
    
    # For each purple: check for Red or Grey nearby
    for p_idx, p_cluster in enumerate(purple):
        p_cent = p_cluster['centroid']
        
        # Look for nearest red
        best_red = None
        best_red_dist = proximity_threshold
        best_red_idx = None
        for r_idx, r_cluster in enumerate(red):
            dist = np.sqrt((r_cluster['centroid'][0] - p_cent[0])**2 +
                          (r_cluster['centroid'][1] - p_cent[1])**2)
            if dist < best_red_dist:
                best_red_dist = dist
                best_red = r_cluster
                best_red_idx = r_idx
        
        # Look for nearest grey
        best_gray = None
        best_gray_dist = proximity_threshold
        best_gray_idx = None
        for g_idx, g_cluster in enumerate(gray):
            dist = np.sqrt((g_cluster['centroid'][0] - p_cent[0])**2 +
                          (g_cluster['centroid'][1] - p_cent[1])**2)
            if dist < best_gray_dist:
                best_gray_dist = dist
                best_gray = g_cluster
                best_gray_idx = g_idx
        
        # Count transitions
        if best_red:
            hole_count += 1
            used_red.add(best_red_idx)
            if best_gray and best_gray_idx not in used_gray:
                # Could be Purple→Red→Grey, but we already counted Purple→Red
                used_gray.add(best_gray_idx)
        elif best_gray:
            hole_count += 1
            used_gray.add(best_gray_idx)
    
    # For remaining red: check for grey nearby
    for r_idx, r_cluster in enumerate(red):
        if r_idx in used_red:
            continue
        
        r_cent = r_cluster['centroid']
        
        best_gray = None
        best_gray_dist = proximity_threshold
        best_gray_idx = None
        for g_idx, g_cluster in enumerate(gray):
            if g_idx in used_gray:
                continue
            dist = np.sqrt((g_cluster['centroid'][0] - r_cent[0])**2 +
                          (g_cluster['centroid'][1] - r_cent[1])**2)
            if dist < best_gray_dist:
                best_gray_dist = dist
                best_gray = g_cluster
                best_gray_idx = g_idx
        
        if best_gray:
            hole_count += 1
            used_gray.add(best_gray_idx)

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"{'='*60}\n")

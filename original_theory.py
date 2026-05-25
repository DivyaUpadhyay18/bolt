import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"ORIGINAL COLOR CLUSTER THEORY")
print(f"Multi-color detection (2+ colors in spatial proximity)")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

def get_color_clusters(frame):
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

def find_nearest(centroid, clusters, threshold):
    if not clusters:
        return None
    best = None
    best_dist = threshold
    for c in clusters:
        dist = np.sqrt((c['centroid'][0] - centroid[0])**2 + 
                       (c['centroid'][1] - centroid[1])**2)
        if dist < best_dist:
            best_dist = dist
            best = c
    return best

# Track holes across frames
hole_tracks = {}  # hole_id -> {'centroid': (x,y), 'colors': set(), 'last_frame': frame_num}
next_hole_id = 1
hole_count = 0
proximity_threshold = 50
max_frame_gap = 5

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... ({hole_count} holes)")
    
    cc = get_color_clusters(frame)
    p_clusters = cc['purple']
    r_clusters = cc['red']
    g_clusters = cc['gray']
    
    # Find multi-color locations this frame
    current_multicolors = []
    
    # For each purple cluster, check if red or gray nearby
    for p in p_clusters:
        r_match = find_nearest(p['centroid'], r_clusters, proximity_threshold)
        g_match = find_nearest(p['centroid'], g_clusters, proximity_threshold)
        
        # Count as multi-color if 2+ colors
        color_count = 1  # purple
        if r_match:
            color_count += 1
        if g_match:
            color_count += 1
        
        if color_count >= 2:
            # This is a valid hole location
            colors_present = {'P'}
            if r_match:
                colors_present.add('R')
            if g_match:
                colors_present.add('G')
            
            avg_x = p['centroid'][0]
            avg_y = p['centroid'][1]
            if r_match:
                avg_x = (avg_x + r_match['centroid'][0]) / 2
                avg_y = (avg_y + r_match['centroid'][1]) / 2
            if g_match:
                avg_x = (avg_x + g_match['centroid'][0]) / 2
                avg_y = (avg_y + g_match['centroid'][1]) / 2
            
            current_multicolors.append({
                'centroid': (int(avg_x), int(avg_y)),
                'colors': colors_present
            })
    
    # Also check for R-G without purple
    for r in r_clusters:
        p_match = find_nearest(r['centroid'], p_clusters, proximity_threshold)
        if not p_match:  # No purple nearby
            g_match = find_nearest(r['centroid'], g_clusters, proximity_threshold)
            if g_match:  # Red and Gray together = 2 colors
                avg_x = (r['centroid'][0] + g_match['centroid'][0]) / 2
                avg_y = (r['centroid'][1] + g_match['centroid'][1]) / 2
                current_multicolors.append({
                    'centroid': (int(avg_x), int(avg_y)),
                    'colors': {'R', 'G'}
                })
    
    # Link current multi-colors to existing tracks
    matched_holes = set()
    
    for mc in current_multicolors:
        # Find closest existing hole track
        best_hole_id = None
        best_dist = proximity_threshold
        
        for hid, track in hole_tracks.items():
            if hid in matched_holes:
                continue
            if frame_idx - track['last_frame'] > max_frame_gap:
                continue  # Track is too old
            
            d = np.sqrt((mc['centroid'][0] - track['centroid'][0])**2 +
                       (mc['centroid'][1] - track['centroid'][1])**2)
            if d < best_dist:
                best_dist = d
                best_hole_id = hid
        
        if best_hole_id is not None:
            # Link to existing hole
            hole_tracks[best_hole_id]['centroid'] = mc['centroid']
            hole_tracks[best_hole_id]['colors'].update(mc['colors'])
            hole_tracks[best_hole_id]['last_frame'] = frame_idx
            matched_holes.add(best_hole_id)
        else:
            # Create new hole track
            hole_tracks[next_hole_id] = {
                'centroid': mc['centroid'],
                'colors': mc['colors'].copy(),
                'last_frame': frame_idx
            }
            matched_holes.add(next_hole_id)
            next_hole_id += 1
    
    # Age out tracks that haven't been updated
    expired_ids = []
    for hid, track in hole_tracks.items():
        if frame_idx - track['last_frame'] > max_frame_gap:
            if hid not in matched_holes:
                expired_ids.append(hid)
    
    for hid in expired_ids:
        hole_count += 1
        del hole_tracks[hid]

# Count remaining active tracks
hole_count += len(hole_tracks)

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"Method: Multi-color clusters with temporal tracking")
print(f"{'='*60}\n")

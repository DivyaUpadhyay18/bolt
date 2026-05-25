import cv2
import numpy as np
from collections import defaultdict

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"TEMPORAL TRACKER - Proper Multi-Color Hole Detection")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

def get_color_clusters(frame, kernel_size=12):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    color_clusters = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
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

# Track multi-color groups across frames
tracks = {}  # track_id -> {'centroid': (x,y), 'colors': set, 'age': frames}
next_id = 1
hole_count = 0
prox = 80
max_age = 5

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
    
    # Find multi-color locations (spatial grouping)
    multicolor_centers = []
    
    for p in p_clusters:
        r_match = find_nearest(p['centroid'], r_clusters, prox)
        g_match = find_nearest(p['centroid'], g_clusters, prox)
        
        if r_match and g_match:
            # P-R-G: average position
            avg_x = (p['centroid'][0] + r_match['centroid'][0] + g_match['centroid'][0]) / 3
            avg_y = (p['centroid'][1] + r_match['centroid'][1] + g_match['centroid'][1]) / 3
            multicolor_centers.append(((int(avg_x), int(avg_y)), {'P', 'R', 'G'}))
        elif r_match:
            # P-R: average position
            avg_x = (p['centroid'][0] + r_match['centroid'][0]) / 2
            avg_y = (p['centroid'][1] + r_match['centroid'][1]) / 2
            multicolor_centers.append(((int(avg_x), int(avg_y)), {'P', 'R'}))
        elif g_match:
            # P-G: average position
            avg_x = (p['centroid'][0] + g_match['centroid'][0]) / 2
            avg_y = (p['centroid'][1] + g_match['centroid'][1]) / 2
            multicolor_centers.append(((int(avg_x), int(avg_y)), {'P', 'G'}))
    
    # Also check R-G without purple
    for r in r_clusters:
        p_match = find_nearest(r['centroid'], p_clusters, prox)
        if not p_match:
            g_match = find_nearest(r['centroid'], g_clusters, prox)
            if g_match:
                avg_x = (r['centroid'][0] + g_match['centroid'][0]) / 2
                avg_y = (r['centroid'][1] + g_match['centroid'][1]) / 2
                multicolor_centers.append(((int(avg_x), int(avg_y)), {'R', 'G'}))
    
    # Link to existing tracks or create new ones
    matched_ids = set()
    new_centers = []
    
    for center, colors in multicolor_centers:
        best_id = None
        best_dist = prox
        
        for tid, track in tracks.items():
            if tid in matched_ids:
                continue
            d = np.sqrt((center[0] - track['centroid'][0])**2 + 
                       (center[1] - track['centroid'][1])**2)
            if d < best_dist:
                best_dist = d
                best_id = tid
        
        if best_id is not None:
            # Link to existing track
            tracks[best_id]['centroid'] = center
            tracks[best_id]['colors'].update(colors)
            tracks[best_id]['age'] = 0
            matched_ids.add(best_id)
        else:
            # Create new track
            new_centers.append((center, colors, next_id))
            next_id += 1
    
    # Age out old tracks
    expired = []
    for tid, track in tracks.items():
        if tid not in matched_ids:
            track['age'] += 1
            if track['age'] > max_age:
                # Count this hole if it had 2+ colors
                if len(track['colors']) >= 2:
                    hole_count += 1
                expired.append(tid)
    
    for tid in expired:
        del tracks[tid]
    
    # Add new tracks
    for center, colors, new_id in new_centers:
        tracks[new_id] = {'centroid': center, 'colors': colors, 'age': 0}

# Count remaining tracks with 2+ colors
for tid, track in tracks.items():
    if len(track['colors']) >= 2:
        hole_count += 1

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"Detection: 2+ colors | Kernel: 12x12 | Proximity: 80px")
print(f"{'='*60}\n")

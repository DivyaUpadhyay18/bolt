import cv2
import numpy as np
from collections import defaultdict

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"IMPROVED DETECTOR - Larger Kernel")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

def get_color_clusters(frame, kernel_size=12):  # Increased from 6 to 12
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    color_clusters = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Larger kernel to better connect nearby pixels
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

# Track holes - use larger proximity threshold for grouped clusters
hole_tracks = defaultdict(lambda: {'colors': set(), 'frame_count': 0, 'last_frame': 0})
hole_count = 0
track_id = 0
prox = 80  # Increased from 50 to 80

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
    
    # Find multi-color clusters
    for p in p_clusters:
        r_match = find_nearest(p['centroid'], r_clusters, prox)
        g_match = find_nearest(p['centroid'], g_clusters, prox)
        
        if r_match and g_match:
            # P-R-G triple (strongest signal)
            sig = ('prg', round(p['centroid'][0]/10), round(p['centroid'][1]/10))
            if sig not in hole_tracks:
                hole_count += 1
            hole_tracks[sig]['colors'].add('P')
            hole_tracks[sig]['colors'].add('R')
            hole_tracks[sig]['colors'].add('G')
            hole_tracks[sig]['last_frame'] = frame_idx
            
        elif r_match:
            # P-R pair
            sig = ('pr', round(p['centroid'][0]/10), round(p['centroid'][1]/10))
            if sig not in hole_tracks:
                hole_count += 1
            hole_tracks[sig]['colors'].add('P')
            hole_tracks[sig]['colors'].add('R')
            hole_tracks[sig]['last_frame'] = frame_idx
            
        elif g_match:
            # P-G pair
            sig = ('pg', round(p['centroid'][0]/10), round(p['centroid'][1]/10))
            if sig not in hole_tracks:
                hole_count += 1
            hole_tracks[sig]['colors'].add('P')
            hole_tracks[sig]['colors'].add('G')
            hole_tracks[sig]['last_frame'] = frame_idx
    
    # Also check for R-G without purple
    for r in r_clusters:
        p_match = find_nearest(r['centroid'], p_clusters, prox)
        if not p_match:  # Only if no purple nearby
            g_match = find_nearest(r['centroid'], g_clusters, prox)
            if g_match:
                sig = ('rg', round(r['centroid'][0]/10), round(r['centroid'][1]/10))
                if sig not in hole_tracks:
                    hole_count += 1
                hole_tracks[sig]['colors'].add('R')
                hole_tracks[sig]['colors'].add('G')
                hole_tracks[sig]['last_frame'] = frame_idx

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"Kernel: 12x12 | Proximity: 80px | Multi-color required")
print(f"{'='*60}\n")

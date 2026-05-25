import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FIXED SEQUENTIAL DETECTOR")
print(f"Sequences: P→R→G | P→R | R→G | P→G (no single colors)")
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

# Track sequences - using a simpler approach
# Key insight: track COMPLETE sequences, not intermediate paths
# Each purple cluster gets ONE p_id for lifetime
# Each red cluster gets ONE r_id for lifetime

purple_tracks = {}  # p_id -> {'centroid': (x,y), 'frame': frame_num, 'completed': False}
red_tracks = {}     # r_id -> {'centroid': (x,y), 'frame': frame_num, 'completed': False}

# Store sequence detection results
detected_sequences = {}  # (seq_type, track_id) -> frame_num

hole_count = 0
next_p_id = 1
next_r_id = 1
prox = 50

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
    
    # STEP 1: Link purple clusters to existing tracks
    p_matched = set()
    for p_cluster in p_clusters:
        best_p_id = None
        best_dist = prox
        
        # Find closest purple track
        for p_id, p_track in purple_tracks.items():
            if p_track['completed']:
                continue  # Don't extend completed tracks
            
            d = np.sqrt((p_cluster['centroid'][0] - p_track['centroid'][0])**2 +
                       (p_cluster['centroid'][1] - p_track['centroid'][1])**2)
            if d < best_dist:
                best_dist = d
                best_p_id = p_id
        
        if best_p_id is not None:
            # Link to existing track
            purple_tracks[best_p_id]['centroid'] = p_cluster['centroid']
            purple_tracks[best_p_id]['frame'] = frame_idx
            p_matched.add(best_p_id)
        else:
            # Create new purple track
            purple_tracks[next_p_id] = {'centroid': p_cluster['centroid'], 'frame': frame_idx, 'completed': False}
            p_matched.add(next_p_id)
            next_p_id += 1
    
    # STEP 2: Link red clusters to existing tracks (and create new ones)
    r_matched = set()
    for r_cluster in r_clusters:
        best_r_id = None
        best_dist = prox
        
        # Find closest red track
        for r_id, r_track in red_tracks.items():
            if r_track['completed']:
                continue  # Don't extend completed tracks
            
            d = np.sqrt((r_cluster['centroid'][0] - r_track['centroid'][0])**2 +
                       (r_cluster['centroid'][1] - r_track['centroid'][1])**2)
            if d < best_dist:
                best_dist = d
                best_r_id = r_id
        
        if best_r_id is not None:
            # Link to existing track
            red_tracks[best_r_id]['centroid'] = r_cluster['centroid']
            red_tracks[best_r_id]['frame'] = frame_idx
            r_matched.add(best_r_id)
        else:
            # Create new red track
            red_tracks[next_r_id] = {'centroid': r_cluster['centroid'], 'frame': frame_idx, 'completed': False}
            r_matched.add(next_r_id)
            next_r_id += 1
    
    # STEP 3: Check for color transitions
    for p_id, p_track in list(purple_tracks.items()):
        if p_track['completed']:
            continue
        
        p_cent = p_track['centroid']
        
        # Check P→R
        r_match = find_nearest(p_cent, r_clusters, prox)
        if r_match:
            # Find which red_id this matches
            best_r_id = None
            best_dist = prox
            for r_id in r_matched:
                r_track = red_tracks[r_id]
                d = np.sqrt((r_track['centroid'][0] - p_cent[0])**2 +
                           (r_track['centroid'][1] - p_cent[1])**2)
                if d < best_dist:
                    best_dist = d
                    best_r_id = r_id
            
            if best_r_id is not None:
                # Check if this combo already detected
                sig = ('pr', p_id, best_r_id)
                if sig not in detected_sequences:
                    detected_sequences[sig] = frame_idx
                    hole_count += 1
                    purple_tracks[p_id]['completed'] = True
                    red_tracks[best_r_id]['completed'] = True
            continue
        
        # Check P→G
        g_match = find_nearest(p_cent, g_clusters, prox)
        if g_match:
            sig = ('pg', p_id)
            if sig not in detected_sequences:
                detected_sequences[sig] = frame_idx
                hole_count += 1
                purple_tracks[p_id]['completed'] = True
            continue
    
    # Check R→G (only for red tracks not already in P→R)
    for r_id, r_track in list(red_tracks.items()):
        if r_track['completed']:
            continue
        
        r_cent = r_track['centroid']
        
        g_match = find_nearest(r_cent, g_clusters, prox)
        if g_match:
            sig = ('rg', r_id)
            if sig not in detected_sequences:
                detected_sequences[sig] = frame_idx
                hole_count += 1
                red_tracks[r_id]['completed'] = True
            continue

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"Sequences counted: P→R, P→G, R→G")
print(f"{'='*60}\n")

import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"REFINED SEQUENTIAL DETECTOR")
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

# Track sequences
active_p = {}  # purple paths
active_r = {}  # red paths (for R→G)
pending_pr = {}  # P→R waiting for G

counted = set()  # Already counted hole signatures
hole_count = 0
path_id = 0
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
    
    # Check P→R waiting for G
    pr_remove = []
    for pr_id, pr_data in list(pending_pr.items()):
        r_cent = pr_data['r_centroid']
        age = frame_idx - pr_data['frame']
        
        # Look for Grey
        g_match = find_nearest(r_cent, g_clusters, prox)
        if g_match:
            # P→R→G complete
            sig = ('prg', pr_id)
            if sig not in counted:
                hole_count += 1
                counted.add(sig)
            pr_remove.append(pr_id)
        elif age > 2:
            # Timeout - already counted as P→R, just remove
            pr_remove.append(pr_id)
    
    for pid in pr_remove:
        del pending_pr[pid]
    
    # Update Purple paths
    p_remove = []
    for p_id, p_data in list(active_p.items()):
        p_cent = p_data['centroid']
        age = frame_idx - p_data['frame']
        
        if age > 5:
            p_remove.append(p_id)
            continue
        
        # Look for Red
        r_match = find_nearest(p_cent, r_clusters, prox)
        if r_match:
            # P→R found - count and add to pending
            sig = ('pr', p_id)
            if sig not in counted:
                hole_count += 1
                counted.add(sig)
            
            # Add to pending for P→R→G upgrade
            pr_id = (frame_idx, p_id)
            pending_pr[pr_id] = {
                'r_centroid': r_match['centroid'],
                'frame': frame_idx
            }
            p_remove.append(p_id)
            continue
        
        # Look for Grey
        g_match = find_nearest(p_cent, g_clusters, prox)
        if g_match:
            # P→G
            sig = ('pg', p_id)
            if sig not in counted:
                hole_count += 1
                counted.add(sig)
            p_remove.append(p_id)
            continue
        
        # Continue with Purple
        p_match = find_nearest(p_cent, r_clusters + p_clusters, prox)
        if p_match:
            active_p[p_id]['centroid'] = p_match['centroid']
            active_p[p_id]['frame'] = frame_idx
    
    for pid in p_remove:
        del active_p[pid]
    
    # Update Red paths (for R→G)
    r_remove = []
    for r_id, r_data in list(active_r.items()):
        r_cent = r_data['centroid']
        age = frame_idx - r_data['frame']
        
        if age > 5:
            r_remove.append(r_id)
            continue
        
        # Look for Grey
        g_match = find_nearest(r_cent, g_clusters, prox)
        if g_match:
            # R→G
            sig = ('rg', r_id)
            if sig not in counted:
                hole_count += 1
                counted.add(sig)
            r_remove.append(r_id)
            continue
        
        # Continue with Red
        r_match = find_nearest(r_cent, r_clusters, prox)
        if r_match:
            active_r[r_id]['centroid'] = r_match['centroid']
            active_r[r_id]['frame'] = frame_idx
    
    for rid in r_remove:
        del active_r[rid]
    
    # Start new Purple paths
    for p_cluster in p_clusters:
        near_p = False
        for p_data in active_p.values():
            d = np.sqrt((p_cluster['centroid'][0] - p_data['centroid'][0])**2 +
                       (p_cluster['centroid'][1] - p_data['centroid'][1])**2)
            if d < prox:
                near_p = True
                break
        
        if not near_p:
            path_id += 1
            active_p[path_id] = {'centroid': p_cluster['centroid'], 'frame': frame_idx}
    
    # Start new Red paths (for R→G only if not in pending)
    for r_cluster in r_clusters:
        near_r = False
        for r_data in active_r.values():
            d = np.sqrt((r_cluster['centroid'][0] - r_data['centroid'][0])**2 +
                       (r_cluster['centroid'][1] - r_data['centroid'][1])**2)
            if d < prox:
                near_r = True
                break
        
        for pr_data in pending_pr.values():
            d = np.sqrt((r_cluster['centroid'][0] - pr_data['r_centroid'][0])**2 +
                       (r_cluster['centroid'][1] - pr_data['r_centroid'][1])**2)
            if d < prox:
                near_r = True
                break
        
        if not near_r:
            path_id += 1
            active_r[path_id] = {'centroid': r_cluster['centroid'], 'frame': frame_idx}

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"Sequences counted: P→R→G, P→R, R→G, P→G")
print(f"{'='*60}\n")

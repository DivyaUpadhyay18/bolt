import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"SEQUENTIAL COLOR TRANSITION DETECTOR - CORRECTED")
print(f"Rules: Purple→Red | Purple→Grey | Purple→Red→Grey | Red→Grey")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

# Track sequences  
active_paths = {}  # Purple paths
pending_pr = {}  # Purple→Red waiting for Grey
counted_sequences = set()  # Track which sequence paths we've counted to avoid double-counting
hole_count = 0
proximity_threshold = 50

def get_color_clusters(frame):
    """Get clusters for each color separately"""
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

def find_nearest_cluster(centroid, clusters, threshold):
    """Find nearest cluster within threshold distance"""
    if not clusters:
        return None
    
    best_match = None
    best_distance = threshold
    
    for cluster in clusters:
        distance = np.sqrt((cluster['centroid'][0] - centroid[0])**2 +
                         (cluster['centroid'][1] - centroid[1])**2)
        if distance < best_distance:
            best_distance = distance
            best_match = cluster
    
    return best_match

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (counted {hole_count} holes so far)")
    
    color_clusters = get_color_clusters(frame)
    
    # Check pending Purple→Red for upgrade to Purple→Red→Grey
    pr_to_remove = []
    for pr_key, pr_data in pending_pr.items():
        last_centroid = pr_data['red_centroid']
        age = frame_idx - pr_data['frame']
        
        # Look for Grey
        grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
        if grey_match:
            # Purple→Red→Grey found
            if pr_key not in counted_sequences:
                hole_count += 1
                counted_sequences.add(pr_key)
            pr_to_remove.append(pr_key)
        elif age > 2:
            # Timeout - count as Purple→Red if not already counted
            if pr_key not in counted_sequences:
                hole_count += 1
                counted_sequences.add(pr_key)
            pr_to_remove.append(pr_key)
    
    for key in pr_to_remove:
        del pending_pr[key]
    
    # Update Purple paths
    p_to_remove = []
    for p_key, p_data in active_paths.items():
        last_centroid = p_data['last_centroid']
        age = frame_idx - p_data['frame']
        
        if age > 5:
            p_to_remove.append(p_key)
            continue
        
        # From Purple, look for Red
        red_match = find_nearest_cluster(last_centroid, color_clusters['red'], proximity_threshold)
        if red_match:
            # Create pending Purple→Red
            pr_key = (frame_idx, id(red_match))
            pending_pr[pr_key] = {
                'red_centroid': red_match['centroid'],
                'frame': frame_idx
            }
            if pr_key not in counted_sequences:
                hole_count += 1  # Count Purple→Red immediately
                counted_sequences.add(pr_key)
            p_to_remove.append(p_key)
            continue
        
        # From Purple, look for Grey
        grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
        if grey_match:
            # Purple→Grey
            pg_key = (frame_idx, id(grey_match), 'pg')
            if pg_key not in counted_sequences:
                hole_count += 1
                counted_sequences.add(pg_key)
            p_to_remove.append(p_key)
            continue
        
        # Continue with purple
        purple_match = find_nearest_cluster(last_centroid, color_clusters['purple'], proximity_threshold)
        if purple_match:
            p_data['last_centroid'] = purple_match['centroid']
            p_data['frame'] = frame_idx
    
    for key in p_to_remove:
        del active_paths[key]
    
    # Start new paths from Red (for Red→Grey)
    for red_cluster in color_clusters['red']:
        # Check if already in a Purple sequence
        is_in_path = False
        for p_data in active_paths.values():
            dist = np.sqrt((red_cluster['centroid'][0] - p_data['last_centroid'][0])**2 +
                          (red_cluster['centroid'][1] - p_data['last_centroid'][1])**2)
            if dist < proximity_threshold:
                is_in_path = True
                break
        
        if not is_in_path:
            # Check for Grey
            grey_match = find_nearest_cluster(red_cluster['centroid'], color_clusters['gray'], proximity_threshold)
            if grey_match:
                # Red→Grey
                rg_key = (frame_idx, id(red_cluster), id(grey_match), 'rg')
                if rg_key not in counted_sequences:
                    hole_count += 1
                    counted_sequences.add(rg_key)
    
    # Start new paths from Purple
    for purple_cluster in color_clusters['purple']:
        is_near_existing = False
        for p_data in active_paths.values():
            distance = np.sqrt((purple_cluster['centroid'][0] - p_data['last_centroid'][0])**2 +
                             (purple_cluster['centroid'][1] - p_data['last_centroid'][1])**2)
            if distance < proximity_threshold:
                is_near_existing = True
                break
        
        if not is_near_existing:
            p_key = (frame_idx, id(purple_cluster))
            active_paths[p_key] = {
                'last_centroid': purple_cluster['centroid'],
                'frame': frame_idx
            }

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"{'='*60}\n")

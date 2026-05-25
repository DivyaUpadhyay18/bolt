import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"SEQUENTIAL COLOR TRANSITION DETECTOR - SIMPLE COUNT")
print(f"Rules: Purple→Red | Purple→Grey | Red→Grey")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

# Track sequences that finished - to avoid counting same path twice
finished_sequences = set()
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

# Track paths: list of (centroid, colors, last_frame, path_id)
active_paths = {}
next_path_id = 1

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (counted {hole_count} holes so far)")
    
    color_clusters = get_color_clusters(frame)
    
    # Update paths
    remove_paths = []
    for path_id, (centroid, colors, last_frame) in list(active_paths.items()):
        age = frame_idx - last_frame
        if age > 5:
            remove_paths.append(path_id)
            continue
        
        last_color = colors[-1]
        
        # Purple → Red or Grey
        if last_color == 'purple':
            red = find_nearest_cluster(centroid, color_clusters['red'], proximity_threshold)
            if red:
                hole_count += 1
                remove_paths.append(path_id)
                continue
            
            grey = find_nearest_cluster(centroid, color_clusters['gray'], proximity_threshold)
            if grey:
                hole_count += 1
                remove_paths.append(path_id)
                continue
            
            purple = find_nearest_cluster(centroid, color_clusters['purple'], proximity_threshold)
            if purple:
                active_paths[path_id] = (purple['centroid'], colors + ['purple'], frame_idx)
        
        # Red → Grey
        elif last_color == 'red':
            grey = find_nearest_cluster(centroid, color_clusters['gray'], proximity_threshold)
            if grey:
                hole_count += 1
                remove_paths.append(path_id)
            else:
                remove_paths.append(path_id)
    
    for pid in remove_paths:
        del active_paths[pid]
    
    # Start from purple
    for p_cluster in color_clusters['purple']:
        near_existing = False
        for (centroid, colors, _) in active_paths.values():
            dist = np.sqrt((p_cluster['centroid'][0] - centroid[0])**2 +
                          (p_cluster['centroid'][1] - centroid[1])**2)
            if dist < proximity_threshold:
                near_existing = True
                break
        
        if not near_existing:
            active_paths[next_path_id] = (p_cluster['centroid'], ['purple'], frame_idx)
            next_path_id += 1

cap.release()

print(f"\n{'='*60}")
print(f"RESULT: {hole_count} HOLES")
print(f"{'='*60}\n")

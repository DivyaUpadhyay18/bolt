import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"SEQUENTIAL COLOR TRANSITION DETECTOR v3 - FIXED")
print(f"Rules: Purple→Red COUNT | Purple→Grey COUNT | Purple→Red→Grey COUNT | Red→Grey COUNT")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

# Track sequences
active_paths = {}  # For Purple sequences
pending_purple_red = {}  # Purple→Red waiting to see if Grey follows
completed_holes = set()
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
    
    # Process pending Purple→Red sequences (waiting for Grey to upgrade to Purple→Red→Grey)
    pending_to_remove = []
    for pending_key, pending_data in pending_purple_red.items():
        last_centroid = pending_data['last_centroid']
        frame_age = frame_idx - pending_data['start_frame']
        
        # Look for Grey to complete Purple→Red→Grey
        grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
        if grey_match:
            # Upgrade Purple→Red to Purple→Red→Grey
            hole_sig = ('purple_red_grey',)
            if hole_sig not in completed_holes:
                hole_count += 1
                completed_holes.add(hole_sig)
            pending_to_remove.append(pending_key)
        elif frame_age > 3:
            # Timeout - this Purple→Red was already counted, just remove pending
            pending_to_remove.append(pending_key)
    
    for key in pending_to_remove:
        del pending_purple_red[key]
    
    # Update active purple-based paths
    paths_to_remove = []
    for path_key, path_data in active_paths.items():
        last_centroid = path_data['last_centroid']
        last_color = path_data['colors'][-1]
        frame_age = frame_idx - path_data['last_frame']
        
        if frame_age > 5:
            paths_to_remove.append(path_key)
            continue
        
        # From purple: look for red or grey
        if last_color == 'purple':
            red_match = find_nearest_cluster(last_centroid, color_clusters['red'], proximity_threshold)
            if red_match:
                # Purple→Red: COUNT immediately and add to pending for possible upgrade
                hole_sig = ('purple_red',)
                if hole_sig not in completed_holes:
                    hole_count += 1
                    completed_holes.add(hole_sig)
                
                # Add to pending to see if it becomes Purple→Red→Grey
                pending_key = (frame_idx, id(red_match))
                pending_purple_red[pending_key] = {
                    'last_centroid': red_match['centroid'],
                    'start_frame': frame_idx
                }
                paths_to_remove.append(path_key)
                continue
            
            grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
            if grey_match:
                # Purple→Grey: COUNT immediately
                hole_sig = ('purple_grey',)
                if hole_sig not in completed_holes:
                    hole_count += 1
                    completed_holes.add(hole_sig)
                paths_to_remove.append(path_key)
                continue
            
            # Continue looking for purple
            purple_match = find_nearest_cluster(last_centroid, color_clusters['purple'], proximity_threshold)
            if purple_match:
                path_data['colors'].append('purple')
                path_data['last_centroid'] = purple_match['centroid']
                path_data['last_frame'] = frame_idx
        
        # From red: look for grey
        elif last_color == 'red':
            grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
            if grey_match:
                # Red→Grey: COUNT immediately
                hole_sig = ('red_grey',)
                if hole_sig not in completed_holes:
                    hole_count += 1
                    completed_holes.add(hole_sig)
                paths_to_remove.append(path_key)
            else:
                # Dead end
                paths_to_remove.append(path_key)
    
    for path_key in paths_to_remove:
        del active_paths[path_key]
    
    # Start new paths from purple clusters
    for purple_cluster in color_clusters['purple']:
        is_near_existing = False
        for path_data in active_paths.values():
            distance = np.sqrt((purple_cluster['centroid'][0] - path_data['last_centroid'][0])**2 +
                             (purple_cluster['centroid'][1] - path_data['last_centroid'][1])**2)
            if distance < proximity_threshold:
                is_near_existing = True
                break
        
        for pending_data in pending_purple_red.values():
            distance = np.sqrt((purple_cluster['centroid'][0] - pending_data['last_centroid'][0])**2 +
                             (purple_cluster['centroid'][1] - pending_data['last_centroid'][1])**2)
            if distance < proximity_threshold:
                is_near_existing = True
                break
        
        if not is_near_existing:
            path_key = (frame_idx, id(purple_cluster))
            active_paths[path_key] = {
                'last_centroid': purple_cluster['centroid'],
                'colors': ['purple'],
                'last_frame': frame_idx
            }

cap.release()

print(f"\n{'='*60}")
print(f"SEQUENTIAL DETECTOR v3 (FIXED) RESULT: {hole_count} HOLES")
print(f"{'='*60}\n")

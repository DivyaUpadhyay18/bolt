import cv2
import numpy as np
from collections import defaultdict

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"SEQUENTIAL COLOR TRANSITION DETECTOR")
print(f"Algorithm: Purple → Red/Grey → Count as hole")
print(f"Processing {frame_count} frames...\n")

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

# Track scanning sequences
scanning_sequences = {}  # sequence_id: {'start_frame': int, 'colors': [list of colors in order], 'centroid_path': [(x,y), ...], 'status': 'active/counted/discarded'}
next_seq_id = 1
hole_count = 0
proximity_threshold = 60

def get_color_clusters_and_centroids(frame):
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

frame_buffer = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (counted {hole_count} holes so far)")
    
    color_clusters = get_color_clusters_and_centroids(frame)
    
    # Process active scanning sequences
    sequences_to_remove = []
    
    for seq_id, seq_data in scanning_sequences.items():
        if seq_data['status'] == 'counted' or seq_data['status'] == 'discarded':
            continue
        
        last_centroid = seq_data['centroid_path'][-1]
        last_color = seq_data['colors'][-1]
        
        # Try to continue the sequence
        found_next = False
        
        # From purple: look for red or grey
        if last_color == 'purple':
            # Check for red
            red_match = find_nearest_cluster(last_centroid, color_clusters['red'], proximity_threshold)
            if red_match:
                seq_data['centroid_path'].append(red_match['centroid'])
                seq_data['colors'].append('red')
                seq_data['status'] = 'counted'
                hole_count += 1
                found_next = True
            
            # Check for grey
            if not found_next:
                grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
                if grey_match:
                    seq_data['centroid_path'].append(grey_match['centroid'])
                    seq_data['colors'].append('gray')
                    seq_data['status'] = 'counted'
                    hole_count += 1
                    found_next = True
            
            # Check if only purple or other colors
            if not found_next:
                purple_match = find_nearest_cluster(last_centroid, color_clusters['purple'], proximity_threshold)
                if purple_match and len(seq_data['colors']) == 1:
                    # Only purple, discard
                    seq_data['status'] = 'discarded'
                else:
                    # Try to continue with purple
                    if purple_match:
                        seq_data['centroid_path'].append(purple_match['centroid'])
                        seq_data['colors'].append('purple')
                    else:
                        seq_data['status'] = 'discarded'
        
        # From red: look for grey
        elif last_color == 'red':
            grey_match = find_nearest_cluster(last_centroid, color_clusters['gray'], proximity_threshold)
            if grey_match:
                seq_data['centroid_path'].append(grey_match['centroid'])
                seq_data['colors'].append('gray')
                seq_data['status'] = 'counted'
                hole_count += 1
                found_next = True
            
            if not found_next:
                seq_data['status'] = 'discarded'
        
        # From grey: look for anything (end sequence as counted if we got here with red->grey path)
        elif last_color == 'gray':
            seq_data['status'] = 'counted'
            if seq_data['status'] == 'counted' and seq_data not in [s for s in scanning_sequences.values() if s['status'] == 'counted']:
                hole_count += 1
    
    # Start new sequences from purple clusters
    for purple_cluster in color_clusters['purple']:
        # Check if this purple is near an existing sequence
        is_near_existing = False
        for seq_data in scanning_sequences.values():
            if seq_data['status'] in ['counted', 'discarded']:
                continue
            last_centroid = seq_data['centroid_path'][-1]
            distance = np.sqrt((purple_cluster['centroid'][0] - last_centroid[0])**2 +
                             (purple_cluster['centroid'][1] - last_centroid[1])**2)
            if distance < proximity_threshold:
                is_near_existing = True
                break
        
        if not is_near_existing:
            # Start new sequence
            scanning_sequences[next_seq_id] = {
                'start_frame': frame_idx,
                'colors': ['purple'],
                'centroid_path': [purple_cluster['centroid']],
                'status': 'active'
            }
            next_seq_id += 1

cap.release()

print(f"\n{'='*60}")
print(f"SEQUENTIAL DETECTOR RESULT: {hole_count} HOLES")
print(f"{'='*60}\n")

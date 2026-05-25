import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"DETECTOR F: Optimized Proximity (50px) + Strict Temporal (2+ frames)")
print(f"Processing {frame_count} frames...\n")

hole_tracks = {}
next_hole_id = 1
proximity_threshold = 50  # Tighter than 65px

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

def get_clusters(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    masks = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        masks[color_name] = cv2.inRange(hsv, lower, upper)
    
    all_mask = masks['purple'] | masks['red'] | masks['gray']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    all_mask = cv2.morphologyEx(all_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clusters = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > 50000:
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
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            continue
        
        colors_present = set()
        cluster_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(cluster_mask, [contour], 0, 255, -1)
        
        if cv2.countNonZero(cluster_mask & masks['purple']) > 0:
            colors_present.add('purple')
        if cv2.countNonZero(cluster_mask & masks['red']) > 0:
            colors_present.add('red')
        if cv2.countNonZero(cluster_mask & masks['gray']) > 0:
            colors_present.add('gray')
        
        if len(colors_present) >= 2:
            clusters.append({'centroid': (cx, cy), 'area': area, 'colors': colors_present})
    
    return clusters

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (found {len(hole_tracks)} holes so far)")
    
    clusters = get_clusters(frame)
    
    matched_clusters = set()
    for hole_id, track in hole_tracks.items():
        if not track:
            continue
        
        last_centroid = track[-1]['centroid']
        best_match = None
        best_distance = proximity_threshold
        
        for cluster_idx, cluster in enumerate(clusters):
            if cluster_idx in matched_clusters:
                continue
            
            distance = np.sqrt((cluster['centroid'][0] - last_centroid[0])**2 +
                             (cluster['centroid'][1] - last_centroid[1])**2)
            
            if distance < best_distance:
                best_distance = distance
                best_match = cluster_idx
        
        if best_match is not None:
            hole_tracks[hole_id].append({'frame': frame_idx, 'centroid': clusters[best_match]['centroid'], 'colors': clusters[best_match]['colors']})
            matched_clusters.add(best_match)
    
    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx not in matched_clusters:
            hole_tracks[next_hole_id] = [{'frame': frame_idx, 'centroid': cluster['centroid'], 'colors': cluster['colors']}]
            next_hole_id += 1

cap.release()

# 2+ frames (strict temporal filtering like detector_strict)
valid_holes = {hole_id: track for hole_id, track in hole_tracks.items() if len(track) >= 2}

final_holes = {}
final_id = 1
for hole_id in sorted(valid_holes.keys()):
    final_holes[final_id] = valid_holes[hole_id]
    final_id += 1

print(f"\n{'='*60}")
print(f"DETECTOR F RESULT: {len(final_holes)} HOLES")
print(f"{'='*60}\n")

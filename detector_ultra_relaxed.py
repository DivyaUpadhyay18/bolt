import cv2
import numpy as np
from collections import defaultdict

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames with ULTRA-RELAXED detection...")

# Track holes across frames with temporal information
hole_tracks = {}  # hole_id -> list of (frame_idx, centroid, area, colors)
next_hole_id = 1
proximity_threshold = 80  # Even stricter spatial separation

# EXTREMELY relaxed color ranges - try to catch everything
color_ranges = {
    'purple': [(100, 0, 0), (180, 255, 255)],      # Very wide
    'red': [(0, 20, 20), (40, 255, 255)],          # Very wide
    'gray': [(0, 0, 20), (180, 150, 220)]          # Very wide
}

def get_multi_color_clusters(frame, frame_idx):
    """Extract multi-color clusters with NO minimum area threshold"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create individual color masks
    masks = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        masks[color_name] = cv2.inRange(hsv, lower, upper)
    
    # Combine all masks
    all_mask = masks['purple'] | masks['red'] | masks['gray']
    
    # Morphological closing with small kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    all_mask = cv2.morphologyEx(all_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clusters = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # NO minimum area requirement - accept everything
        if area < 1:
            continue
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Determine which colors this cluster contains
        colors_present = set()
        cluster_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(cluster_mask, [contour], 0, 255, -1)
        
        if cv2.countNonZero(cluster_mask & masks['purple']) > 0:
            colors_present.add('purple')
        if cv2.countNonZero(cluster_mask & masks['red']) > 0:
            colors_present.add('red')
        if cv2.countNonZero(cluster_mask & masks['gray']) > 0:
            colors_present.add('gray')
        
        # Require at least 1 color present (no multi-color requirement!)
        if len(colors_present) >= 1:
            clusters.append({
                'centroid': (cx, cy),
                'area': area,
                'colors': colors_present,
                'contour': contour
            })
    
    return clusters

# Process all frames
for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (found {len(hole_tracks)} holes so far)")
    
    clusters = get_multi_color_clusters(frame, frame_idx)
    
    # Match clusters to existing tracks
    matched_clusters = set()
    for hole_id, track in hole_tracks.items():
        if not track:  # Empty track
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
            hole_tracks[hole_id].append({
                'frame': frame_idx,
                'centroid': clusters[best_match]['centroid'],
                'area': clusters[best_match]['area'],
                'colors': clusters[best_match]['colors']
            })
            matched_clusters.add(best_match)
    
    # Create new tracks for unmatched clusters
    for cluster_idx, cluster in enumerate(clusters):
        if cluster_idx not in matched_clusters:
            hole_tracks[next_hole_id] = [{
                'frame': frame_idx,
                'centroid': cluster['centroid'],
                'area': cluster['area'],
                'colors': cluster['colors']
            }]
            next_hole_id += 1

cap.release()

# Filter holes: keep those with at least 1 frame presence (ultra-relaxed)
valid_holes = {}
for hole_id, track in hole_tracks.items():
    if len(track) >= 1:
        valid_holes[hole_id] = track

# Assign final numbering
final_holes = {}
final_id = 1
for hole_id in sorted(valid_holes.keys()):
    if hole_id in valid_holes:
        final_holes[final_id] = valid_holes[hole_id]
        final_id += 1

print("\n" + "="*60)
print(f"BOLT HOLES DETECTED: {len(final_holes)}")
print("="*60 + "\n")

for hole_num in sorted(final_holes.keys()):
    track = final_holes[hole_num]
    frames = [p['frame'] for p in track]
    min_frame = min(frames)
    max_frame = max(frames)
    num_frames = len(track)
    
    all_colors = set()
    for point in track:
        all_colors.update(point['colors'])
    color_str = ', '.join(sorted(all_colors))
    
    print(f"BH-{hole_num}: Frames {min_frame}-{max_frame} ({num_frames} frames), Colors: {color_str}")

print("\n" + "="*60)
print(f"FINAL ANSWER: {len(final_holes)}")
print("="*60)

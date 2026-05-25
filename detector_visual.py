import cv2
import numpy as np
from collections import defaultdict

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames - detecting YELLOW RING structures...")

# Track holes across frames
hole_tracks = {}
next_hole_id = 1
proximity_threshold = 60

def get_ring_structures(frame, frame_idx):
    """
    Detect yellow ring structures (bolt holes) with dark centers.
    Yellow rings are the key visual signature in B-scan images.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yellow in HSV: H around 25-35, high S, high V
    # But also include orange-yellow (H 20-40) to catch variations
    yellow_low = np.array([15, 100, 100], dtype=np.uint8)
    yellow_high = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
    
    # Dark areas (centers of holes): low V
    dark_low = np.array([0, 0, 0], dtype=np.uint8)
    dark_high = np.array([180, 255, 80], dtype=np.uint8)
    dark_mask = cv2.inRange(hsv, dark_low, dark_high)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in yellow mask
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rings = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Ring-like structures have moderate area
        if area < 50 or area > 5000:
            continue
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Get bounding box for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            continue
        
        aspect_ratio = w / h
        
        # Ring/ellipse should be roughly circular to oval (0.5-2.0 ratio)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        
        # Check if there's dark center (indication of bolt hole)
        center_region = dark_mask[max(0, cy-10):min(dark_mask.shape[0], cy+10),
                                   max(0, cx-10):min(dark_mask.shape[1], cx+10)]
        
        dark_pixel_count = cv2.countNonZero(center_region)
        
        # Should have some dark pixels in center region (indication of hole)
        if dark_pixel_count < 5:
            continue
        
        rings.append({
            'centroid': (cx, cy),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'contour': contour
        })
    
    return rings

# Process all frames
for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{frame_count}... (found {len(hole_tracks)} holes so far)")
    
    rings = get_ring_structures(frame, frame_idx)
    
    # Match rings to existing tracks (proximity-based)
    matched_rings = set()
    for hole_id, track in hole_tracks.items():
        if not track:
            continue
        
        last_centroid = track[-1]['centroid']
        best_match = None
        best_distance = proximity_threshold
        
        for ring_idx, ring in enumerate(rings):
            if ring_idx in matched_rings:
                continue
            
            distance = np.sqrt((ring['centroid'][0] - last_centroid[0])**2 +
                             (ring['centroid'][1] - last_centroid[1])**2)
            
            if distance < best_distance:
                best_distance = distance
                best_match = ring_idx
        
        if best_match is not None:
            hole_tracks[hole_id].append({
                'frame': frame_idx,
                'centroid': rings[best_match]['centroid'],
                'area': rings[best_match]['area']
            })
            matched_rings.add(best_match)
    
    # Create new tracks for unmatched rings
    for ring_idx, ring in enumerate(rings):
        if ring_idx not in matched_rings:
            hole_tracks[next_hole_id] = [{
                'frame': frame_idx,
                'centroid': ring['centroid'],
                'area': ring['area']
            }]
            next_hole_id += 1

cap.release()

# Filter: keep holes appearing in 3+ frames
valid_holes = {}
for hole_id, track in hole_tracks.items():
    if len(track) >= 3:
        valid_holes[hole_id] = track

# Final numbering
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
    avg_area = np.mean([p['area'] for p in track])
    
    print(f"BH-{hole_num}: Frames {min_frame}-{max_frame} ({num_frames} frames), Avg Area: {avg_area:.0f}")

print("\n" + "="*60)
print(f"FINAL ANSWER: {len(final_holes)}")
print("="*60)

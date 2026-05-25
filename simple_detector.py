import cv2
import numpy as np
from collections import defaultdict
import os

VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"

def process_video_ultra_simple():
    """Ultra-simple and fast bolt hole detection"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video with {total_frames} frames...")
    
    bolt_holes = {}
    next_id = 1
    frame_num = 0
    
    BOTTOM_BLUE = 641
    RED_LINE = 639
    DETECTION_ZONE_TOP = BOTTOM_BLUE - 5
    
    while True:
        ret, frame = self.cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get masks
        purple = cv2.inRange(hsv, np.array([125, 30, 30]), np.array([155, 255, 255]))
        red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red |= cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        gray = cv2.inRange(hsv, (15, 0, 80), (45, 100, 200))
        
        # Find contours for each color
        holes_found = []
        
        for color_name, mask in [('purple', purple), ('red', red), ('gray', gray)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Ignore very small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Check if in detection zone
                    if DETECTION_ZONE_TOP <= center_y <= RED_LINE:
                        holes_found.append({
                            'color': color_name,
                            'center': (center_x, center_y),
                            'bbox': (x, y, x+w, y+h),
                            'area': area
                        })
        
        # Group holes by proximity (multi-color clusters)
        if len(holes_found) >= 2:
            used = set()
            for i, hole_i in enumerate(holes_found):
                if i in used:
                    continue
                
                cluster = [hole_i]
                used.add(i)
                
                for j, hole_j in enumerate(holes_found):
                    if j <= i or j in used:
                        continue
                    
                    dist = np.sqrt((hole_i['center'][0] - hole_j['center'][0])**2 + 
                                 (hole_i['center'][1] - hole_j['center'][1])**2)
                    
                    if dist < 100:  # Merge if close
                        cluster.append(hole_j)
                        used.add(j)
                
                # Check if multi-color
                colors = set([h['color'] for h in cluster])
                if len(colors) >= 2:
                    # Check color order
                    color_order = {'purple': 0, 'red': 1, 'gray': 2}
                    color_positions = {}
                    for color in colors:
                        positions = [h['center'][0] for h in cluster if h['color'] == color]
                        color_positions[color] = min(positions)
                    
                    order_ok = True
                    for c1 in colors:
                        for c2 in colors:
                            if color_order[c1] < color_order[c2]:
                                if color_positions[c1] > color_positions[c2]:
                                    order_ok = False
                    
                    if order_ok:
                        # This is a valid bolt hole
                        all_centers = [h['center'] for h in cluster]
                        avg_center = (int(np.mean([c[0] for c in all_centers])),
                                    int(np.mean([c[1] for c in all_centers])))
                        
                        # Try to match with existing hole
                        matched = False
                        for hid, hdata in bolt_holes.items():
                            if not hdata['active']:
                                continue
                            last_center = hdata['last_center']
                            dist = np.sqrt((avg_center[0] - last_center[0])**2 + 
                                         (avg_center[1] - last_center[1])**2)
                            if dist < 80:
                                hdata['frames'].append(frame_num)
                                hdata['last_center'] = avg_center
                                matched = True
                                break
                        
                        if not matched:
                            bolt_holes[next_id] = {
                                'frames': [frame_num],
                                'last_center': avg_center,
                                'active': True,
                                'colors': colors
                            }
                            next_id += 1
        
        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total_frames}...")
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"BOLT HOLES DETECTED: {len(bolt_holes)}")
    for hid in sorted(bolt_holes.keys()):
        frames = bolt_holes[hid]['frames']
        print(f"  BH-{hid}: Frames {frames[0]}-{frames[-1]} ({len(frames)} frames)")
    print(f"{'='*60}\n")
    
    return len(bolt_holes)

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video with {total_frames} frames...")
    
    bolt_holes = {}
    next_id = 1
    frame_num = 0
    
    BOTTOM_BLUE = 641
    RED_LINE = 639
    DETECTION_ZONE_TOP = BOTTOM_BLUE - 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get masks
        purple = cv2.inRange(hsv, np.array([125, 30, 30]), np.array([155, 255, 255]))
        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask |= cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        gray = cv2.inRange(hsv, (15, 0, 80), (45, 100, 200))
        
        # Find contours for each color
        holes_found = []
        
        for color_name, mask in [('purple', purple), ('red', red_mask), ('gray', gray)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Ignore very small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Check if in detection zone
                    if DETECTION_ZONE_TOP <= center_y <= RED_LINE:
                        holes_found.append({
                            'color': color_name,
                            'center': (center_x, center_y),
                            'bbox': (x, y, x+w, y+h),
                            'area': area
                        })
        
        # Group holes by proximity (multi-color clusters)
        if len(holes_found) >= 2:
            used = set()
            for i, hole_i in enumerate(holes_found):
                if i in used:
                    continue
                
                cluster = [hole_i]
                used.add(i)
                
                for j, hole_j in enumerate(holes_found):
                    if j <= i or j in used:
                        continue
                    
                    dist = np.sqrt((hole_i['center'][0] - hole_j['center'][0])**2 + 
                                 (hole_i['center'][1] - hole_j['center'][1])**2)
                    
                    if dist < 100:  # Merge if close
                        cluster.append(hole_j)
                        used.add(j)
                
                # Check if multi-color
                colors = set([h['color'] for h in cluster])
                if len(colors) >= 2:
                    # Check color order
                    color_order = {'purple': 0, 'red': 1, 'gray': 2}
                    color_positions = {}
                    for color in colors:
                        positions = [h['center'][0] for h in cluster if h['color'] == color]
                        color_positions[color] = min(positions)
                    
                    order_ok = True
                    for c1 in colors:
                        for c2 in colors:
                            if color_order[c1] < color_order[c2]:
                                if color_positions[c1] > color_positions[c2]:
                                    order_ok = False
                    
                    if order_ok:
                        # This is a valid bolt hole
                        all_centers = [h['center'] for h in cluster]
                        avg_center = (int(np.mean([c[0] for c in all_centers])),
                                    int(np.mean([c[1] for c in all_centers])))
                        
                        # Try to match with existing hole
                        matched = False
                        for hid, hdata in bolt_holes.items():
                            if not hdata['active']:
                                continue
                            last_center = hdata['last_center']
                            dist = np.sqrt((avg_center[0] - last_center[0])**2 + 
                                         (avg_center[1] - last_center[1])**2)
                            if dist < 80:
                                hdata['frames'].append(frame_num)
                                hdata['last_center'] = avg_center
                                matched = True
                                break
                        
                        if not matched:
                            bolt_holes[next_id] = {
                                'frames': [frame_num],
                                'last_center': avg_center,
                                'active': True,
                                'colors': colors
                            }
                            next_id += 1
        
        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total_frames}...")
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"BOLT HOLES DETECTED: {len(bolt_holes)}")
    for hid in sorted(bolt_holes.keys()):
        frames = bolt_holes[hid]['frames']
        print(f"  BH-{hid}: Frames {frames[0]}-{frames[-1]} ({len(frames)} frames)")
    print(f"{'='*60}\n")
    
    print(f"FINAL: {len(bolt_holes)}")

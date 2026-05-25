import cv2
import numpy as np
from collections import defaultdict
import os

VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"

def detect_bolt_holes():
    """Detect bolt holes by finding multi-color clusters"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {total_frames} frames...")
    
    bolt_holes = {}
    next_id = 1
    frame_num = 0
    
    # Looser color ranges
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges with LOOSE constraints to find all color pixels
        purple_mask = cv2.inRange(hsv, (100, 0, 0), (170, 255, 255))
        red_mask = cv2.inRange(hsv, (0, 30, 30), (40, 255, 255))
        gray_mask = cv2.inRange(hsv, (0, 0, 30), (180, 150, 200))
        
        # Combine all colors
        all_mask = cv2.bitwise_or(cv2.bitwise_or(purple_mask, red_mask), gray_mask)
        
        # Apply morphological closing to connect nearby colors
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        all_mask = cv2.morphologyEx(all_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        holes_found = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small clusters
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check if cluster contains multiple colors
            region_purple = cv2.inRange(hsv[max(0, y-5):min(frame.shape[0], y+h+5), 
                                               max(0, x-5):min(frame.shape[1], x+w+5)],
                                       (100, 0, 0), (170, 255, 255))
            region_red = cv2.inRange(hsv[max(0, y-5):min(frame.shape[0], y+h+5), 
                                           max(0, x-5):min(frame.shape[1], x+w+5)],
                                     (0, 30, 30), (40, 255, 255))
            region_gray = cv2.inRange(hsv[max(0, y-5):min(frame.shape[0], y+h+5), 
                                            max(0, x-5):min(frame.shape[1], x+w+5)],
                                      (0, 0, 30), (180, 150, 200))
            
            p_count = np.sum(region_purple > 0)
            r_count = np.sum(region_red > 0)
            g_count = np.sum(region_gray > 0)
            
            colors_present = []
            if p_count > 2:
                colors_present.append('purple')
            if r_count > 2:
                colors_present.append('red')
            if g_count > 2:
                colors_present.append('gray')
            
            # Valid bolt hole needs at least 2 colors
            if len(colors_present) >= 2:
                # Check color order: purple < red < gray (left to right)
                # For simplicity, accept any 2-color or 3-color combo
                holes_found.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, x+w, y+h),
                    'colors': colors_present,
                    'area': area
                })
        
        # Track holes across frames
        for hole in holes_found:
            center = hole['center']
            matched = False
            
            for hid, hdata in bolt_holes.items():
                if not hdata.get('active', True):
                    continue
                
                last_center = hdata.get('last_center', center)
                dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                # If close to existing hole, it's the same one
                if dist < 100:
                    hdata['frames'].append(frame_num)
                    hdata['last_center'] = center
                    matched = True
                    break
            
            if not matched:
                bolt_holes[next_id] = {
                    'frames': [frame_num],
                    'last_center': center,
                    'colors': hole['colors'],
                    'active': True
                }
                next_id += 1
        
        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total_frames}... (found {len(bolt_holes)} holes so far)")
    
    cap.release()
    
    return bolt_holes

def main():
    holes = detect_bolt_holes()
    
    print(f"\n{'='*60}")
    print(f"BOLT HOLES DETECTED: {len(holes)}")
    print(f"{'='*60}\n")
    
    if len(holes) > 0:
        for hid in sorted(holes.keys()):
            frames = holes[hid]['frames']
            print(f"BH-{hid}: Frames {frames[0]}-{frames[-1]} ({len(frames)} frames), Colors: {', '.join(holes[hid]['colors'])}")
    
    print(f"\n{'='*60}\n")
    print(f"FINAL ANSWER: {len(holes)}")

if __name__ == "__main__":
    main()

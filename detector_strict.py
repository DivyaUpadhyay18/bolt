import cv2
import numpy as np
from collections import defaultdict
import os

VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"

def detect_bolt_holes_strict():
    """Strict bolt hole detection with better spatial separation"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {total_frames} frames with STRICT detection...")
    
    bolt_holes = {}
    next_id = 1
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # STRICTER color ranges - more selective
        purple_mask = cv2.inRange(hsv, (120, 20, 20), (160, 255, 255))
        red_mask = cv2.inRange(hsv, (0, 60, 60), (25, 255, 255))
        gray_mask = cv2.inRange(hsv, (10, 10, 60), (60, 120, 200))
        
        # Combine
        all_mask = cv2.bitwise_or(cv2.bitwise_or(purple_mask, red_mask), gray_mask)
        
        # Less aggressive morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        all_mask = cv2.morphologyEx(all_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        holes_found = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # STRICTER size requirements
            if area < 200 or area > 50000:  # Tighter bounds
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio - holes should be roughly circular/oval
            aspect_ratio = float(w) / (h + 0.001)
            if aspect_ratio > 4 or aspect_ratio < 0.25:
                continue
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check if cluster has REQUIRED multi-color composition
            region = hsv[max(0, y-2):min(frame.shape[0], y+h+2), 
                          max(0, x-2):min(frame.shape[1], x+w+2)]
            
            p_count = np.sum(cv2.inRange(region, (120, 20, 20), (160, 255, 255)) > 0)
            r_count = np.sum(cv2.inRange(region, (0, 60, 60), (25, 255, 255)) > 0)
            g_count = np.sum(cv2.inRange(region, (10, 10, 60), (60, 120, 200)) > 0)
            
            total_colored = p_count + r_count + g_count
            
            # Require minimum threshold of each color
            colors_present = []
            if p_count > total_colored * 0.15:  # At least 15% purple
                colors_present.append('purple')
            if r_count > total_colored * 0.05:  # At least 5% red
                colors_present.append('red')
            if g_count > total_colored * 0.15:  # At least 15% gray
                colors_present.append('gray')
            
            # REQUIRE at least 2 colors
            if len(colors_present) < 2:
                continue
            
            holes_found.append({
                'center': (center_x, center_y),
                'bbox': (x, y, x+w, y+h),
                'colors': colors_present,
                'area': area,
                'width': w,
                'height': h
            })
        
        # Track holes with STRICTER spatial matching
        for hole in holes_found:
            center = hole['center']
            matched = False
            
            # Look for CLOSEST existing hole, but only if very close
            best_match = None
            best_dist = float('inf')
            
            for hid, hdata in bolt_holes.items():
                if not hdata.get('active', True):
                    continue
                
                last_center = hdata.get('last_center', center)
                
                # STRICT distance threshold - holes must be very close
                dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if dist < best_dist and dist < 50:  # STRICTER threshold: 50 pixels
                    best_match = hid
                    best_dist = dist
            
            if best_match is not None:
                bolt_holes[best_match]['frames'].append(frame_num)
                bolt_holes[best_match]['last_center'] = center
                bolt_holes[best_match]['bbox'] = hole['bbox']
                matched = True
            
            if not matched:
                # NEW hole - check it doesn't overlap with existing holes
                is_new = True
                for hid, hdata in bolt_holes.items():
                    if hdata.get('active', False):
                        last_center = hdata.get('last_center', (0, 0))
                        dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                        # If close to an active hole but not matched, mark it
                        if dist < 30:
                            is_new = False
                            break
                
                if is_new:
                    bolt_holes[next_id] = {
                        'frames': [frame_num],
                        'last_center': center,
                        'colors': hole['colors'],
                        'active': True,
                        'first_frame': frame_num
                    }
                    next_id += 1
        
        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total_frames}... (found {len(bolt_holes)} holes so far)")
    
    cap.release()
    
    # Clean up - remove holes that only appeared once or very briefly
    filtered_holes = {}
    for hid, hdata in bolt_holes.items():
        if len(hdata['frames']) >= 3:  # Must appear in at least 3 frames
            filtered_holes[hid] = hdata
    
    return filtered_holes

def main():
    holes = detect_bolt_holes_strict()
    
    print(f"\n{'='*60}")
    print(f"BOLT HOLES DETECTED: {len(holes)}")
    print(f"{'='*60}\n")
    
    if len(holes) > 0:
        for hid in sorted(holes.keys()):
            frames = holes[hid]['frames']
            frame_range = f"{frames[0]}-{frames[-1]}"
            num_frames = len(frames)
            print(f"BH-{hid}: Frames {frame_range} ({num_frames} frames), Colors: {', '.join(holes[hid]['colors'])}")
    
    print(f"\n{'='*60}\n")
    print(f"FINAL ANSWER: {len(holes)}")

if __name__ == "__main__":
    main()

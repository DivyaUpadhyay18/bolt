import cv2
import numpy as np
import os

SAMPLE_DIR = r"c:\Users\divya\Downloads\bolt\samples"
frame_path = os.path.join(SAMPLE_DIR, "sample_frame_1000.jpg")
frame = cv2.imread(frame_path)

if frame is None:
    print(f"Could not load frame: {frame_path}")
else:
    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}\n")
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges with relaxed constraints
    colors = {
        'purple': ([110, 0, 0], [180, 255, 255]),
        'red': ([0, 50, 50], [30, 255, 255]),
        'gray': ([0, 0, 50], [180, 100, 200])
    }
    
    # Find contours for each color across the entire frame
    for color_name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Apply morphology to connect nearby pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"{color_name.upper()}:")
        print(f"  Total pixels: {np.sum(mask > 0)}")
        print(f"  Number of contours: {len(contours)}")
        
        # Sort by area
        contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Show top contours
        for idx, contour in enumerate(contours_by_area[:10]):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            print(f"    #{idx}: area={area:.0f}, pos=({x},{y}), size=({w}x{h})")
        print()
    
    # Now look for multi-color clusters by analyzing regions
    print("LOOKING FOR MULTI-COLOR CLUSTERS:\n")
    
    # Define looser ranges for initial detection
    purple_mask = cv2.inRange(hsv, (100, 0, 0), (170, 255, 255))
    red_mask = cv2.inRange(hsv, (0, 30, 30), (40, 255, 255))
    gray_mask = cv2.inRange(hsv, (0, 0, 30), (180, 150, 200))
    
    # Find clusters
    all_mask = cv2.bitwise_or(cv2.bitwise_or(purple_mask, red_mask), gray_mask)
    
    # Apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    all_mask = cv2.morphologyEx(all_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total combined multi-color clusters: {len(contours)}")
    print(f"Total colored pixels: {np.sum(all_mask > 0)}\n")
    
    # Analyze clusters with significant size
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10:  # Only clusters with area > 10
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check what colors are in this cluster
            region = frame[max(0, y-5):min(height, y+h+5), max(0, x-5):min(width, x+w+5)]
            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            p_count = np.sum(cv2.inRange(region_hsv, (100, 0, 0), (170, 255, 255)) > 0)
            r_count = np.sum(cv2.inRange(region_hsv, (0, 30, 30), (40, 255, 255)) > 0)
            g_count = np.sum(cv2.inRange(region_hsv, (0, 0, 30), (180, 150, 200)) > 0)
            
            colors_in_cluster = []
            if p_count > 0:
                colors_in_cluster.append('P')
            if r_count > 0:
                colors_in_cluster.append('R')
            if g_count > 0:
                colors_in_cluster.append('G')
            
            print(f"Cluster {idx}: area={area:.0f}, pos=({center_x},{center_y}), size=({w}x{h}), colors={','.join(colors_in_cluster)}")

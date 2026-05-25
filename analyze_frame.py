import cv2
import numpy as np
import os

# Analyze specific frames in detail
SAMPLE_DIR = r"c:\Users\divya\Downloads\bolt\samples"

frame_path = os.path.join(SAMPLE_DIR, "sample_frame_1000.jpg")
frame = cv2.imread(frame_path)

if frame is None:
    print(f"Could not load frame: {frame_path}")
else:
    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Analyze detection zone (around y=640, between 635-645)
    detection_zone = frame[630:650, :]
    detection_zone_hsv = cv2.cvtColor(detection_zone, cv2.COLOR_BGR2HSV)
    
    print("\n=== Full Frame Analysis ===")
    
    # Test different color ranges
    ranges_to_test = {
        'purple_125_155': ([125, 30, 30], [155, 255, 255]),
        'purple_120_160': ([120, 20, 20], [160, 255, 255]),
        'red_0_10': ([0, 100, 100], [10, 255, 255]),
        'red_170_180': ([170, 100, 100], [180, 255, 255]),
        'gray_15_45': ([15, 0, 80], [45, 100, 200]),
        'gray_10_50': ([10, 0, 50], [50, 200, 255]),
    }
    
    for name, (lower, upper) in ranges_to_test.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = np.sum(mask > 0)
        if count > 0:
            print(f"  {name}: {count} pixels")
    
    print("\n=== Detection Zone (y=630-650) ===")
    detection_zone_hsv = hsv[630:650, :]
    
    for name, (lower, upper) in ranges_to_test.items():
        mask = cv2.inRange(detection_zone_hsv, np.array(lower), np.array(upper))
        count = np.sum(mask > 0)
        if count > 0:
            print(f"  {name}: {count} pixels")
    
    # Find contours in detection zone for each color
    print("\n=== Contours in Detection Zone ===")
    for color_name in ['purple_120_160', 'red_0_10', 'red_170_180', 'gray_10_50']:
        lower, upper = ranges_to_test[color_name]
        mask = cv2.inRange(detection_zone_hsv, np.array(lower), np.array(upper))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            print(f"\n  {color_name}:")
            for idx, contour in enumerate(contours[:5]):  # Show first 5
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                print(f"    Contour {idx}: area={area:.1f}, bbox=({x},{y},{w},{h})")
    
    # Save visualization
    debug_frame = frame.copy()
    
    # Draw detection zone
    cv2.line(debug_frame, (0, 630), (width, 630), (0, 255, 0), 2)  # Detection zone top
    cv2.line(debug_frame, (0, 650), (width, 650), (0, 0, 255), 2)  # Detection zone bottom (red line)
    
    # Highlight colors in detection zone
    lower_p = np.array([120, 20, 20])
    upper_p = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv[630:650], lower_p, upper_p)
    
    lower_r1 = np.array([0, 100, 100])
    upper_r1 = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv[630:650], lower_r1, upper_r1)
    
    lower_g = np.array([10, 0, 50])
    upper_g = np.array([50, 200, 255])
    gray_mask = cv2.inRange(hsv[630:650], lower_g, upper_g)
    
    # Mark detected colors
    purple_pts = np.where(purple_mask > 0)
    red_pts = np.where(red_mask > 0)
    gray_pts = np.where(gray_mask > 0)
    
    for py, px in list(zip(purple_pts[1], purple_pts[0]))[:100]:  # Limit to 100 points
        cv2.circle(debug_frame, (px, 630 + py), 2, (255, 0, 255), -1)
    
    for ry, rx in list(zip(red_pts[1], red_pts[0]))[:100]:
        cv2.circle(debug_frame, (rx, 630 + ry), 2, (0, 0, 255), -1)
    
    for gy, gx in list(zip(gray_pts[1], gray_pts[0]))[:100]:
        cv2.circle(debug_frame, (gx, 630 + gy), 2, (128, 128, 128), -1)
    
    output_path = os.path.join(SAMPLE_DIR, "debug_analysis.jpg")
    cv2.imwrite(output_path, debug_frame)
    print(f"\nDebug frame saved: {output_path}")

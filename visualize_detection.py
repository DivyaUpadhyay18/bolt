import cv2
import numpy as np

video_path = r'c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

color_ranges = {
    'purple': [(100, 5, 5), (175, 255, 255)],
    'red': [(0, 30, 30), (40, 255, 255)],
    'gray': [(0, 0, 30), (180, 150, 200)]
}

def get_color_clusters(frame):
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
            
            clusters.append({'centroid': (cx, cy), 'area': area, 'bbox': (x,y,w,h)})
        
        color_clusters[color_name] = clusters
    
    return color_clusters

# Examine specific frames
test_frames = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

for test_frame in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    cc = get_color_clusters(frame)
    p_count = len(cc['purple'])
    r_count = len(cc['red'])
    g_count = len(cc['gray'])
    
    print(f"\nFrame {test_frame}:")
    print(f"  Purple clusters: {p_count}")
    print(f"  Red clusters:    {r_count}")
    print(f"  Gray clusters:   {g_count}")
    
    # Show which clusters are multi-color (2+ colors)
    for p in cc['purple']:
        p_near_r = any(np.sqrt((p['centroid'][0]-r['centroid'][0])**2 + (p['centroid'][1]-r['centroid'][1])**2) < 50 
                       for r in cc['red'])
        p_near_g = any(np.sqrt((p['centroid'][0]-g['centroid'][0])**2 + (p['centroid'][1]-g['centroid'][1])**2) < 50 
                       for g in cc['gray'])
        if p_near_r or p_near_g:
            colors = []
            if p_near_r: colors.append('R')
            if p_near_g: colors.append('G')
            print(f"    P at {p['centroid']} near: {colors}")

cap.release()
print("\nAnalysis complete.")

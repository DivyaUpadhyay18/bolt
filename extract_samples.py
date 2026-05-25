import cv2
import numpy as np
import os

VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"
SAMPLE_DIR = r"c:\Users\divya\Downloads\bolt\samples"

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

os.makedirs(SAMPLE_DIR, exist_ok=True)

# Extract sample frames
frame_numbers = [100, 500, 1000, 2000, 3000, 4000, 5000]

for frame_num in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        out_path = os.path.join(SAMPLE_DIR, f"sample_frame_{frame_num}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Extracted frame {frame_num}")
        
        # Analyze colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        purple_mask = cv2.inRange(hsv, np.array([125, 30, 30]), np.array([155, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        gray_mask = cv2.inRange(hsv, np.array([15, 0, 80]), np.array([45, 100, 200]))
        
        purple_count = np.sum(purple_mask > 0)
        red_count = np.sum(red_mask > 0)
        gray_count = np.sum(gray_mask > 0)
        
        print(f"  Purple: {purple_count} pixels, Red: {red_count} pixels, Gray: {gray_count} pixels")

cap.release()
print("Sample extraction complete!")

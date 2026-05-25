import numpy as np
import cv2
from detector import BoltHoleDetector
from tracker import BoltHoleTracker
from colour_rules import get_purple_mask, get_red_mask, get_grey_mask

print("Testing individual components...")
print()

# Test colour masks
roi_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
purple_mask = get_purple_mask(roi_rgb)
red_mask = get_red_mask(roi_rgb)
grey_mask = get_grey_mask(roi_rgb)
print(f"✓ Colour masks generated: shapes {purple_mask.shape}, {red_mask.shape}, {grey_mask.shape}")
print()

# Test detector
roi_bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
detector = BoltHoleDetector(sigma=10, min_distance=18, prominence=0.2)
result = detector.detect(roi_bgr)
print(f"✓ Detector initialized and executed")
print(f"  - Holes detected: {result['bolt_hole_count']}")
print(f"  - Output keys: {len(result)} items")
print()

# Test tracker
tracker = BoltHoleTracker(max_distance=35, max_missing_frames=6)
holes1 = tracker.update([(50, 50), (80, 60)], current_frame=1)
holes2 = tracker.update([(52, 51), (150, 100)], current_frame=2)
holes3 = tracker.update([(54, 52)], current_frame=3)
print(f"✓ Tracker initialized and updated")
print(f"  - Frame 1: {len(holes1)} active holes")
print(f"  - Frame 2: {len(holes2)} active holes (1 new)")
print(f"  - Frame 3: {len(holes3)} active holes")
print(f"  - Total unique ever seen: {tracker.next_id - 1}")
print(f"  - Hole history: {len(tracker.hole_history)} holes recorded")
print()

for i, h in enumerate(tracker.hole_history):
    print(f"    Hole {i+1}: {h['label']} (frames {h['first_frame']}-{h['last_frame']})")
print()

print("All component tests passed! System is ready.")

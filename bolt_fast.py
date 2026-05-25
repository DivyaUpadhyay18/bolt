import cv2
import numpy as np
from collections import defaultdict
import os

VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"

class FastBoltHoleDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.bolt_holes = {}
        self.next_hole_id = 1
        self.frame_count = 0
        
        self.bottom_blue_line = 641
        self.red_line = 639
        
    def extract_colors_fast(self, frame):
        """Extract color masks for purple, red, gray"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Purple mask
        purple_mask = cv2.inRange(hsv, np.array([125, 30, 30]), np.array([155, 255, 255]))
        
        # Red masks (two ranges in HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Gray mask
        gray_mask = cv2.inRange(hsv, np.array([15, 0, 80]), np.array([45, 100, 200]))
        
        return purple_mask, red_mask, gray_mask
    
    def find_clusters_in_mask(self, mask):
        """Find connected components in a binary mask"""
        if mask.sum() == 0:
            return []
        
        # Use erosion + dilation to connect nearby pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(processed)
        
        clusters = []
        for label_id in range(1, num_labels):
            pts = np.where(labels == label_id)
            if len(pts[0]) >= 2:
                cluster_points = list(zip(pts[1], pts[0]))  # (x, y)
                clusters.append(np.array(cluster_points, dtype=np.float32))
        
        return clusters
    
    def analyze_frame_fast(self, frame):
        """Fast frame analysis for bolt holes"""
        purple_mask, red_mask, gray_mask = self.extract_colors_fast(frame)
        
        # Get clusters for each color
        purple_clusters = self.find_clusters_in_mask(purple_mask)
        red_clusters = self.find_clusters_in_mask(red_mask)
        gray_clusters = self.find_clusters_in_mask(gray_mask)
        
        all_clusters = []
        for pc in purple_clusters:
            all_clusters.append(('purple', pc))
        for rc in red_clusters:
            all_clusters.append(('red', rc))
        for gc in gray_clusters:
            all_clusters.append(('gray', gc))
        
        if len(all_clusters) < 2:
            return []
        
        # Merge nearby clusters of different colors
        valid_holes = []
        used = set()
        
        for i, (color1, pts1) in enumerate(all_clusters):
            if i in used:
                continue
            
            center1 = np.mean(pts1, axis=0)
            y1 = center1[1]
            
            # Check if in detection zone
            if not (self.bottom_blue_line - 5 <= y1 <= self.red_line):
                used.add(i)
                continue
            
            cluster_dict = {color1: pts1}
            used.add(i)
            
            # Find nearby clusters of other colors
            for j, (color2, pts2) in enumerate(all_clusters):
                if j <= i or j in used or color2 == color1:
                    continue
                
                center2 = np.mean(pts2, axis=0)
                dist = np.linalg.norm(center1 - center2)
                
                # If close enough, consider them part of same bolt hole
                if dist < 80:
                    y2 = center2[1]
                    if (self.bottom_blue_line - 5 <= y2 <= self.red_line):
                        cluster_dict[color2] = pts2
                        used.add(j)
            
            # Check if multi-color
            colors = set(cluster_dict.keys())
            if len(colors) < 2:
                continue
            
            # Check color order
            color_order = {'purple': 0, 'red': 1, 'gray': 2}
            positions = {}
            for color, pts in cluster_dict.items():
                positions[color] = np.min(pts[:, 0])
            
            order_ok = True
            for c1 in colors:
                for c2 in colors:
                    if color_order[c1] < color_order[c2]:
                        if positions[c1] > positions[c2]:
                            order_ok = False
                            break
                if not order_ok:
                    break
            
            if not order_ok:
                continue
            
            # Combine all points
            all_pts = np.vstack([cluster_dict[c] for c in cluster_dict])
            
            # Check coherence
            x_range = np.max(all_pts[:, 0]) - np.min(all_pts[:, 0])
            y_range = np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])
            
            if x_range < 5 and y_range < 5:
                continue
            
            x_min, x_max = int(np.min(all_pts[:, 0])), int(np.max(all_pts[:, 0]))
            y_min, y_max = int(np.min(all_pts[:, 1])), int(np.max(all_pts[:, 1]))
            
            valid_holes.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'center': (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
                'colors': colors
            })
        
        return valid_holes
    
    def track_holes(self, frame_holes):
        """Track holes across frames"""
        for hole in frame_holes:
            center = hole['center']
            matched = False
            
            for hole_id, hole_data in self.bolt_holes.items():
                if not hole_data['active']:
                    continue
                
                last_center = hole_data['last_center']
                dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if dist < 60:
                    hole_data['frames'].append(self.frame_count)
                    hole_data['last_center'] = center
                    hole_data['bbox'] = hole['bbox']
                    matched = True
                    break
            
            if not matched:
                self.bolt_holes[self.next_hole_id] = {
                    'frames': [self.frame_count],
                    'last_center': center,
                    'active': True,
                    'bbox': hole['bbox'],
                    'colors': hole['colors'],
                    'first_frame': self.frame_count
                }
                self.next_hole_id += 1
    
    def process_video_fast(self):
        """Process video quickly"""
        print(f"Processing: {os.path.basename(self.video_path)}")
        print(f"Frames: {self.total_frames}, Size: {self.frame_width}x{self.frame_height}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            holes = self.analyze_frame_fast(frame)
            if holes:
                self.track_holes(holes)
            
            self.frame_count += 1
            
            if self.frame_count % 500 == 0:
                print(f"  {self.frame_count}/{self.total_frames}...")
        
        self.cap.release()
        print(f"Complete. {self.total_frames} frames processed.")
    
    def get_results(self):
        """Get final results"""
        total = len(self.bolt_holes)
        print(f"\n{'='*60}")
        print(f"BOLT HOLES DETECTED: {total}")
        print(f"{'='*60}")
        
        if total > 0:
            print(f"\nBolt hole details:")
            for hid in sorted(self.bolt_holes.keys()):
                hdata = self.bolt_holes[hid]
                frames = hdata['frames']
                print(f"  BH-{hid}: Frames {frames[0]}-{frames[-1]} ({len(frames)} frames)")
        
        print(f"\n{'='*60}\n")
        return total

def main():
    detector = FastBoltHoleDetector(VIDEO_PATH)
    detector.process_video_fast()
    total = detector.get_results()
    print(f"FINAL: {total}")

if __name__ == "__main__":
    main()

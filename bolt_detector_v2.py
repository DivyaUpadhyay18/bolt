import cv2
import numpy as np
from collections import defaultdict
import os
from sklearn.cluster import DBSCAN

# Configuration
VIDEO_PATH = r"c:\Users\divya\Downloads\bolt\SRT_BScan - v5.8.5 - [B-Scan] 2026-05-20 12-12-19 (1).mp4"
OUTPUT_DIR = r"c:\Users\divya\Downloads\bolt\output"

class BoltHoleDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.bolt_holes = {}  # {hole_id: {frames, positions, colors, first_frame}}
        self.next_hole_id = 1
        self.frame_count = 0
        self.detection_results = defaultdict(list)
        
        self.top_blue_line = None
        self.bottom_blue_line = None
        self.red_line = None
        
    def detect_reference_lines(self, frame):
        """Detect the two blue lines and red line"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect blue lines
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Detect red lines
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find blue lines
        blue_coords = np.where(blue_mask > 0)
        if len(blue_coords[0]) > 0:
            blue_y_values = blue_coords[0]
            blue_y_unique = np.unique(blue_y_values)
            if len(blue_y_unique) >= 2:
                self.top_blue_line = int(np.min(blue_y_unique))
                self.bottom_blue_line = int(np.max(blue_y_unique))
        
        # Find red line
        red_coords = np.where(red_mask > 0)
        if len(red_coords[0]) > 0:
            self.red_line = int(np.max(red_coords[0]))
    
    def is_in_detection_zone(self, y):
        """Check if y coordinate is in detection zone"""
        if self.bottom_blue_line is None or self.red_line is None:
            return False
        tolerance = 5
        return (self.bottom_blue_line - tolerance) <= y <= self.red_line
    
    def extract_colored_clusters(self, frame):
        """Extract clusters of purple, red, and gray dots"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        clusters = {
            'purple': [],
            'red': [],
            'gray': []
        }
        
        # Purple detection
        lower_p = np.array([125, 30, 30])
        upper_p = np.array([155, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_p, upper_p)
        purple_coords = np.where(purple_mask > 0)
        if len(purple_coords[0]) > 0:
            clusters['purple'] = list(zip(purple_coords[1], purple_coords[0]))
        
        # Red detection
        lower_r1 = np.array([0, 100, 100])
        upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 100, 100])
        upper_r2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)
        red_coords = np.where(red_mask > 0)
        if len(red_coords[0]) > 0:
            clusters['red'] = list(zip(red_coords[1], red_coords[0]))
        
        # Gray detection
        lower_g = np.array([15, 0, 80])
        upper_g = np.array([45, 100, 200])
        gray_mask = cv2.inRange(hsv, lower_g, upper_g)
        gray_coords = np.where(gray_mask > 0)
        if len(gray_coords[0]) > 0:
            clusters['gray'] = list(zip(gray_coords[1], gray_coords[0]))
        
        return clusters
    
    def analyze_clusters(self, frame):
        """Detect and validate bolt holes in frame"""
        clusters_data = self.extract_colored_clusters(frame)
        
        valid_holes = []
        
        # Combine all colored points
        all_points = []
        color_labels = []
        
        for color in ['purple', 'red', 'gray']:
            for p in clusters_data[color]:
                all_points.append(p)
                color_labels.append(color)
        
        if len(all_points) < 3:
            return valid_holes
        
        all_points = np.array(all_points)
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=20, min_samples=2).fit(all_points)
        labels = clustering.labels_
        
        # Process each cluster
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_pts = all_points[cluster_mask]
            cluster_colors = [color_labels[i] for i in range(len(color_labels)) if cluster_mask[i]]
            
            # CONDITION 2: Check minimum 2 colors
            unique_colors = set(cluster_colors)
            target_colors_present = unique_colors & {'purple', 'red', 'gray'}
            
            if len(target_colors_present) < 2:
                continue
            
            # CONDITION 1: Check detection zone
            y_coords = cluster_pts[:, 1]
            avg_y = np.mean(y_coords)
            
            if not self.is_in_detection_zone(avg_y):
                continue
            
            # CONDITION 3: Check color order
            color_positions = {}
            for color in target_colors_present:
                color_pts = cluster_pts[[i for i in range(len(cluster_colors)) if cluster_colors[i] == color]]
                if len(color_pts) > 0:
                    color_positions[color] = np.min(color_pts[:, 0])
            
            # Validate order: Purple < Red < Gray
            order_valid = True
            color_order = {'purple': 0, 'red': 1, 'gray': 2}
            
            for c1 in target_colors_present:
                for c2 in target_colors_present:
                    if color_order[c1] < color_order[c2]:
                        if color_positions[c1] > color_positions[c2]:
                            order_valid = False
                            break
                if not order_valid:
                    break
            
            if not order_valid:
                continue
            
            # CONDITION 4: Check spatial coherence
            if len(cluster_pts) < 3:
                continue
            
            x_coords = cluster_pts[:, 0]
            y_coords = cluster_pts[:, 1]
            
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            
            if x_range < 5 and y_range < 5:
                continue
            
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            
            valid_holes.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'center': (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
                'colors': target_colors_present,
                'points': cluster_pts
            })
        
        return valid_holes
    
    def track_holes(self, frame_holes):
        """Track bolt holes across frames"""
        for hole in frame_holes:
            center = hole['center']
            
            matched = False
            for hole_id, hole_data in self.bolt_holes.items():
                if not hole_data['active']:
                    continue
                
                last_center = hole_data['last_center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if distance < 50:
                    hole_data['frames'].append(self.frame_count)
                    hole_data['positions'].append(center)
                    hole_data['last_center'] = center
                    hole_data['bbox'] = hole['bbox']
                    matched = True
                    break
            
            if not matched:
                self.bolt_holes[self.next_hole_id] = {
                    'frames': [self.frame_count],
                    'positions': [center],
                    'last_center': center,
                    'active': True,
                    'bbox': hole['bbox'],
                    'colors': hole['colors'],
                    'first_frame': self.frame_count
                }
                self.next_hole_id += 1
    
    def process_video(self):
        """Process the entire video"""
        print(f"Processing video: {os.path.basename(self.video_path)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}, Total frames: {self.total_frames}")
        
        # First pass: detect reference lines
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.detect_reference_lines(frame)
            print(f"Reference lines detected:")
            print(f"  Top blue line: y={self.top_blue_line}")
            print(f"  Bottom blue line: y={self.bottom_blue_line}")
            print(f"  Red line: y={self.red_line}")
        
        # Second pass: detect bolt holes
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_holes = self.analyze_clusters(frame)
            if frame_holes:
                self.detection_results[self.frame_count] = frame_holes
                self.track_holes(frame_holes)
            
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                print(f"Processed {self.frame_count}/{self.total_frames} frames...")
        
        self.cap.release()
        print(f"Completed processing all {self.total_frames} frames.")
    
    def generate_output_frames(self):
        """Generate output frames with bolt hole annotations"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self.cap = cv2.VideoCapture(self.video_path)
        frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num in self.detection_results:
                # Draw reference lines
                if self.bottom_blue_line:
                    cv2.line(frame, (0, self.bottom_blue_line), (self.frame_width, self.bottom_blue_line), (255, 0, 0), 2)
                if self.red_line:
                    cv2.line(frame, (0, self.red_line), (self.frame_width, self.red_line), (0, 0, 255), 2)
                
                # Draw bolt holes with bounding boxes
                for hole_id, hole_data in self.bolt_holes.items():
                    if frame_num in hole_data['frames']:
                        x_min, y_min, x_max, y_max = hole_data['bbox']
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, f"BH-{hole_id}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_num:04d}.jpg")
                cv2.imwrite(output_path, frame)
            
            frame_num += 1
        
        self.cap.release()
    
    def get_final_count(self):
        """Get total count of unique bolt holes"""
        return len(self.bolt_holes)
    
    def print_results(self):
        """Print results"""
        total_holes = self.get_final_count()
        print(f"\n{'='*60}")
        print(f"BOLT HOLE DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Bolt holes detected: {total_holes}")
        
        print(f"\nFrames with bolt holes and their numbering:")
        frames_with_holes = sorted(self.detection_results.keys())
        for frame_num in frames_with_holes[:50]:  # Print first 50
            hole_ids = []
            for hole_id, hole_data in self.bolt_holes.items():
                if frame_num in hole_data['frames']:
                    hole_ids.append(f"BH-{hole_id}")
            if hole_ids:
                print(f"  Frame {frame_num}: {', '.join(hole_ids)}")
        
        if len(frames_with_holes) > 50:
            print(f"  ... and {len(frames_with_holes) - 50} more frames with holes")
        
        print(f"\nDetailed bolt hole tracking:")
        for hole_id in sorted(self.bolt_holes.keys()):
            hole_data = self.bolt_holes[hole_id]
            frames = hole_data['frames']
            print(f"  BH-{hole_id}: Frames {frames[0]} to {frames[-1]} ({len(frames)} frames), Colors: {', '.join(hole_data['colors'])}")
        
        print(f"\n{'='*60}\n")

def main():
    detector = BoltHoleDetector(VIDEO_PATH)
    detector.process_video()
    # Uncomment to generate output frames (slows down processing)
    # detector.generate_output_frames()
    detector.print_results()
    
    total = detector.get_final_count()
    print(f"FINAL ANSWER: {total}")

if __name__ == "__main__":
    main()

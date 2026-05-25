import cv2
import numpy as np
from collections import defaultdict
import os
from scipy.ndimage import label

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
        
        self.bolt_holes = {}
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
    
    def extract_colored_points(self, frame):
        """Extract points of each color"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        points = {
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
            points['purple'] = list(zip(purple_coords[1], purple_coords[0]))
        
        # Red detection
        lower_r1 = np.array([0, 100, 100])
        upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 100, 100])
        upper_r2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)
        red_coords = np.where(red_mask > 0)
        if len(red_coords[0]) > 0:
            points['red'] = list(zip(red_coords[1], red_coords[0]))
        
        # Gray detection
        lower_g = np.array([15, 0, 80])
        upper_g = np.array([45, 100, 200])
        gray_mask = cv2.inRange(hsv, lower_g, upper_g)
        gray_coords = np.where(gray_mask > 0)
        if len(gray_coords[0]) > 0:
            points['gray'] = list(zip(gray_coords[1], gray_coords[0]))
        
        return points
    
    def find_connected_components(self, points, distance=25):
        """Find connected components in point cloud using simple distance-based grouping"""
        if len(points) == 0:
            return []
        
        points = np.array(points, dtype=np.float32)
        visited = np.zeros(len(points), dtype=bool)
        components = []
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # BFS to find connected component
            component = [points[i]]
            visited[i] = True
            queue = [i]
            
            while queue:
                current_idx = queue.pop(0)
                current_point = points[current_idx]
                
                # Find all nearby unvisited points
                for j in range(len(points)):
                    if visited[j]:
                        continue
                    
                    dist = np.linalg.norm(current_point - points[j])
                    if dist <= distance:
                        component.append(points[j])
                        visited[j] = True
                        queue.append(j)
            
            if len(component) >= 3:
                components.append(np.array(component))
        
        return components
    
    def analyze_clusters(self, frame):
        """Detect and validate bolt holes in frame"""
        points_data = self.extract_colored_points(frame)
        
        # Find connected components for each color
        components_by_color = {}
        for color, points in points_data.items():
            if len(points) > 0:
                components_by_color[color] = self.find_connected_components(points, distance=25)
        
        valid_holes = []
        
        # Now find multi-color clusters by proximity
        all_components = []
        for color, components in components_by_color.items():
            for comp in components:
                all_components.append((color, comp))
        
        # Group components of different colors that are close
        used = set()
        for i, (color1, comp1) in enumerate(all_components):
            if i in used:
                continue
            
            cluster = {color1: comp1}
            used.add(i)
            
            # Find nearby components of other colors
            center1 = np.mean(comp1, axis=0)
            
            for j, (color2, comp2) in enumerate(all_components):
                if j <= i or j in used or color2 == color1:
                    continue
                
                center2 = np.mean(comp2, axis=0)
                dist = np.linalg.norm(center1 - center2)
                
                if dist < 50:  # Proximity threshold
                    cluster[color2] = comp2
                    used.add(j)
            
            # Check conditions
            colors_present = set(cluster.keys())
            target_colors = colors_present & {'purple', 'red', 'gray'}
            
            # CONDITION 2: At least 2 colors
            if len(target_colors) < 2:
                continue
            
            # Combine all points in cluster
            all_pts = np.vstack([cluster[c] for c in cluster])
            
            # CONDITION 1: In detection zone
            avg_y = np.mean(all_pts[:, 1])
            if not self.is_in_detection_zone(avg_y):
                continue
            
            # CONDITION 3: Color order
            color_order = {'purple': 0, 'red': 1, 'gray': 2}
            color_positions = {}
            for color in target_colors:
                color_positions[color] = np.min(cluster[color][:, 0])
            
            order_valid = True
            for c1 in target_colors:
                for c2 in target_colors:
                    if color_order[c1] < color_order[c2]:
                        if color_positions[c1] > color_positions[c2]:
                            order_valid = False
                            break
                if not order_valid:
                    break
            
            if not order_valid:
                continue
            
            # CONDITION 4: Shape coherence
            if len(all_pts) < 3:
                continue
            
            x_range = np.max(all_pts[:, 0]) - np.min(all_pts[:, 0])
            y_range = np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])
            
            if x_range < 5 and y_range < 5:
                continue
            
            x_min, x_max = int(np.min(all_pts[:, 0])), int(np.max(all_pts[:, 0]))
            y_min, y_max = int(np.min(all_pts[:, 1])), int(np.max(all_pts[:, 1]))
            
            valid_holes.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'center': (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
                'colors': target_colors,
                'points': all_pts
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
                
                if distance < 60:
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
        print(f"Resolution: {self.frame_width}x{self.frame_height}, Total frames: {self.total_frames}")
        
        # Detect reference lines
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.detect_reference_lines(frame)
            print(f"Reference lines: Top={self.top_blue_line}, Bottom={self.bottom_blue_line}, Red={self.red_line}")
        
        # Process all frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_holes = self.analyze_clusters(frame)
            if frame_holes:
                self.track_holes(frame_holes)
            
            self.frame_count += 1
            
            if self.frame_count % 200 == 0:
                print(f"Processed {self.frame_count}/{self.total_frames} frames...")
        
        self.cap.release()
        print(f"Processing complete. {self.total_frames} frames analyzed.")
    
    def print_results(self):
        """Print results"""
        total_holes = len(self.bolt_holes)
        print(f"\n{'='*60}")
        print(f"BOLT HOLE DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"\nBolt holes detected: {total_holes}")
        
        if total_holes > 0:
            print(f"\nDetailed bolt hole information:")
            for hole_id in sorted(self.bolt_holes.keys()):
                hole_data = self.bolt_holes[hole_id]
                frames = hole_data['frames']
                start_frame = frames[0]
                end_frame = frames[-1]
                colors = ', '.join(sorted(hole_data['colors']))
                print(f"  BH-{hole_id}: Frames {start_frame}-{end_frame} ({len(frames)} frames), Colors: {colors}")
        
        print(f"\n{'='*60}\n")

def main():
    detector = BoltHoleDetector(VIDEO_PATH)
    detector.process_video()
    detector.print_results()
    
    print(f"FINAL ANSWER: {len(detector.bolt_holes)}")

if __name__ == "__main__":
    main()

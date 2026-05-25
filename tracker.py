"""
Persistent bolt hole tracking across frames.
Uses greedy distance-based matching.
"""

import numpy as np
from scipy.spatial.distance import cdist


class BoltHoleTracker:
    """
    Tracks bolt holes across frames with persistent ID numbering.
    IDs increment only for new holes, never reset or reused.
    """
    
    def __init__(self, max_distance=35, max_missing_frames=6):
        """
        Initialize tracker.
        
        Args:
            max_distance: Max pixels distance to match hole (same hole)
            max_missing_frames: Retire hole if absent for this many frames
        """
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        
        self.next_id = 1
        self.active_holes = {}
        # { hole_id: {"cx": int, "cy": int, "missing": int,
        #             "first_frame": int, "last_frame": int} }
        
        self.hole_history = []
        # [{"id": int, "label": str, "first_frame": int, "last_frame": int,
        #   "max_cx": int, "max_cy": int}]
    
    def update(self, detected_positions, current_frame):
        """
        Update tracker with new detections.
        
        Args:
            detected_positions: list of (cx, cy) tuples from detector
            current_frame: int frame number
        
        Returns:
            list of active hole dicts
            [{"id": int, "label": str, "cx": int, "cy": int}]
        """
        
        detected_array = np.array(detected_positions) if detected_positions else np.empty((0, 2))
        
        # Compute distances if we have both detections and active holes
        matched_detections = set()
        matched_holes = set()
        
        if len(detected_array) > 0 and len(self.active_holes) > 0:
            active_ids = sorted(self.active_holes.keys())
            active_positions = np.array([
                [self.active_holes[hid]["cx"], self.active_holes[hid]["cy"]]
                for hid in active_ids
            ])
            
            distances = cdist(detected_array, active_positions, metric='euclidean')
            
            # Greedy matching: sort by distance ascending
            matches = []
            for i, d in enumerate(distances):
                for j, dist_val in enumerate(d):
                    matches.append((dist_val, i, j))
            
            matches.sort(key=lambda x: x[0])
            
            for dist_val, det_idx, hole_idx in matches:
                if det_idx in matched_detections or hole_idx in matched_holes:
                    continue
                if dist_val <= self.max_distance:
                    # Match found
                    hole_id = active_ids[hole_idx]
                    cx, cy = int(detected_array[det_idx, 0]), int(detected_array[det_idx, 1])
                    
                    self.active_holes[hole_id]["cx"] = cx
                    self.active_holes[hole_id]["cy"] = cy
                    self.active_holes[hole_id]["missing"] = 0
                    self.active_holes[hole_id]["last_frame"] = current_frame
                    
                    matched_detections.add(det_idx)
                    matched_holes.add(hole_idx)
        
        # Unmatched detections → new holes
        for det_idx in range(len(detected_array)):
            if det_idx not in matched_detections:
                cx, cy = int(detected_array[det_idx, 0]), int(detected_array[det_idx, 1])
                new_id = self.next_id
                self.next_id += 1
                
                self.active_holes[new_id] = {
                    "cx": cx,
                    "cy": cy,
                    "missing": 0,
                    "first_frame": current_frame,
                    "last_frame": current_frame,
                }
                
                self.hole_history.append({
                    "id": new_id,
                    "label": f"BH-{new_id}",
                    "first_frame": current_frame,
                    "last_frame": current_frame,
                    "max_cx": cx,
                    "max_cy": cy,
                })
        
        # Unmatched active holes → increment missing
        for hole_id in self.active_holes.keys():
            # Check if this hole was matched
            hole_idx_in_active = sorted(self.active_holes.keys()).index(hole_id)
            if hole_idx_in_active not in matched_holes:
                self.active_holes[hole_id]["missing"] += 1
        
        # Retire holes with too many missing frames
        holes_to_retire = [
            hole_id for hole_id in self.active_holes.keys()
            if self.active_holes[hole_id]["missing"] > self.max_missing_frames
        ]
        
        for hole_id in holes_to_retire:
            # Update history
            last_frame = self.active_holes[hole_id]["last_frame"]
            for h in self.hole_history:
                if h["id"] == hole_id:
                    h["last_frame"] = last_frame
                    break
            
            del self.active_holes[hole_id]
        
        # Return active holes with labels
        active_list = []
        for hole_id in sorted(self.active_holes.keys()):
            hole = self.active_holes[hole_id]
            active_list.append({
                "id": hole_id,
                "label": f"BH-{hole_id}",
                "cx": hole["cx"],
                "cy": hole["cy"],
            })
        
        return active_list

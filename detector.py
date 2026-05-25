"""
Bolt hole detector using horizontal projection algorithm.
Detects holes based on 4 valid colour combinations.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from colour_rules import get_purple_mask, get_red_mask, get_grey_mask


class BoltHoleDetector:
    """
    Detects bolt holes using horizontal projection + Gaussian smoothing.
    Valid combinations: Purple+Grey, Purple+Red+Grey, Red+Grey, Purple+Red.
    """
    
    def __init__(self, sigma=10, min_distance=18, prominence=0.2):
        """
        Initialize detector.
        
        Args:
            sigma: Gaussian smoothing width (higher = tolerates more scatter)
            min_distance: Minimum pixels between detected holes
            prominence: Peak prominence threshold for peak detection
        """
        self.sigma = sigma
        self.min_distance = min_distance
        self.prominence = prominence
    
    def detect(self, roi_bgr):
        """
        Detect bolt holes in ROI using horizontal projection.
        
        Args:
            roi_bgr: ROI in BGR format (from find_bscan_roi)
        
        Returns:
            dict with detection results and signals
        """
        
        # STEP 1 — Get colour masks
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        purple_mask = get_purple_mask(roi_rgb)
        red_mask = get_red_mask(roi_rgb)
        grey_mask = get_grey_mask(roi_rgb)
        
        # STEP 2 — Vertical projection onto x-axis
        purple_proj = purple_mask.sum(axis=0).astype(float)
        red_proj = red_mask.sum(axis=0).astype(float)
        grey_proj = grey_mask.sum(axis=0).astype(float)
        
        # STEP 3 — Gaussian smoothing
        ps = gaussian_filter1d(purple_proj, sigma=self.sigma)
        rs = gaussian_filter1d(red_proj, sigma=self.sigma)
        gs = gaussian_filter1d(grey_proj, sigma=self.sigma)
        
        # STEP 4 — Four combo signals using np.minimum
        combo1 = np.minimum(ps, gs)  # Purple + Grey
        combo2 = np.minimum(np.minimum(ps, rs), gs)  # Purple + Red + Grey
        combo3 = np.minimum(rs, gs)  # Red + Grey
        combo4 = np.minimum(ps, rs)  # Purple + Red
        combined = np.maximum.reduce([combo1, combo2, combo3, combo4])
        
        # STEP 5 — Find peaks
        threshold = max(combined.mean() * 0.25, 0.2)
        peaks, _ = find_peaks(
            combined,
            distance=self.min_distance,
            height=threshold,
            prominence=self.prominence
        )
        
        # STEP 6 — Get centroid for each peak
        bolt_hole_positions = []
        for peak_x in peaks:
            x0 = max(0, peak_x - 20)
            x1 = min(roi_bgr.shape[1], peak_x + 20)
            band = (purple_mask[:, x0:x1] | red_mask[:, x0:x1] | 
                    grey_mask[:, x0:x1])
            
            if band.any():
                ys, _ = np.where(band)
                cy = int(ys.mean())
            else:
                cy = roi_bgr.shape[0] // 2
            
            bolt_hole_positions.append((int(peak_x), cy))
        
        # STEP 7 — Build annotated ROI
        annotated_roi = roi_bgr.copy()
        annotated_roi[purple_mask] = (180, 0, 180)  # Magenta in BGR
        annotated_roi[red_mask] = (0, 0, 220)  # Red in BGR
        annotated_roi[grey_mask] = (180, 180, 0)  # Cyan in BGR
        
        # Debug print
        print(f"[DETECT] purple={purple_mask.sum()} red={red_mask.sum()} "
              f"grey={grey_mask.sum()} combined_max={combined.max():.3f} "
              f"threshold={threshold:.3f} peaks={len(peaks)}")
        
        return {
            "bolt_hole_count": len(bolt_hole_positions),
            "bolt_hole_positions": bolt_hole_positions,
            "purple_mask": purple_mask,
            "red_mask": red_mask,
            "grey_mask": grey_mask,
            "annotated_roi": annotated_roi,
            "purple_proj": ps,
            "red_proj": rs,
            "grey_proj": gs,
            "combined_signal": combined,
        }

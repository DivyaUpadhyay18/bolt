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
    
    def __init__(self, sigma=10, min_distance=18, prominence=0.1):
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
    
    def detect(self, roi_bgr, sigma=10, min_distance=18, prominence=0.1):
        """
        Detect bolt holes in the B-Scan ROI.
        A bolt hole = 2 or more colour channels present at same x position.
        Single colour alone is never a bolt hole.
        Works for tight clusters AND scattered dot patterns.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return {
                "bolt_hole_count":     0,
                "bolt_hole_positions": [],
                "annotated_roi":       roi_bgr,
                "combined_signal":     np.array([]),
                "purple_proj":         np.array([]),
                "red_proj":            np.array([]),
                "grey_proj":           np.array([]),
            }

        # ── Step 1: Get colour masks ───────────────────────────────
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        purple_mask = get_purple_mask(roi_rgb)
        red_mask    = get_red_mask(roi_rgb)
        grey_mask   = get_grey_mask(roi_rgb)

        # ── Step 2: Vertical projection onto x-axis ───────────────
        # For each x column count pixels of each colour
        purple_proj = purple_mask.sum(axis=0).astype(float)
        red_proj    = red_mask.sum(axis=0).astype(float)
        grey_proj   = grey_mask.sum(axis=0).astype(float)

        # ── Step 3: Gaussian smoothing ────────────────────────────
        # Bridges gaps between scattered dots of the same hole
        # Higher sigma = more tolerant of scatter
        ps = gaussian_filter1d(purple_proj, sigma=sigma)
        rs = gaussian_filter1d(red_proj,    sigma=sigma)
        gs = gaussian_filter1d(grey_proj,   sigma=sigma)

        # ── Step 4: Build combined signal ─────────────────────────
        # Use np.minimum so BOTH channels must be present
        # np.minimum returns 0 if either is 0
        # Take max across all valid two-colour combinations
        # This is HIGH wherever ANY two colours overlap

        combo_pg  = np.minimum(ps, gs)        # Purple + Grey
        combo_pr  = np.minimum(ps, rs)        # Purple + Red
        combo_rg  = np.minimum(rs, gs)        # Red + Grey
        combo_prg = np.minimum(combo_pg, rs)  # Purple + Red + Grey

        combined = np.maximum.reduce([
            combo_pg,
            combo_pr,
            combo_rg,
            combo_prg
        ])

        # ── Step 5: Find peaks in combined signal ─────────────────
        if combined.max() == 0:
            # No two-colour overlap found anywhere
            print(f"[DETECT] combined signal is all zero — "
                  f"purple={purple_proj.sum():.0f} "
                  f"red={red_proj.sum():.0f} "
                  f"grey={grey_proj.sum():.0f}")
            return {
                "bolt_hole_count":     0,
                "bolt_hole_positions": [],
                "annotated_roi":       roi_bgr.copy(),
                "combined_signal":     combined,
                "purple_proj":         ps,
                "red_proj":            rs,
                "grey_proj":           gs,
            }

        # Threshold = 10% of the combined signal maximum
        # Low enough to catch weak/scattered holes
        # High enough to ignore pure noise floor
        threshold = combined.max() * 0.10

        peaks, _ = find_peaks(
            combined,
            distance=min_distance,
            height=threshold,
            prominence=prominence
        )

        print(f"[DETECT] purple_sum={purple_proj.sum():.0f} "
              f"red_sum={red_proj.sum():.0f} "
              f"grey_sum={grey_proj.sum():.0f} "
              f"combined_max={combined.max():.3f} "
              f"threshold={threshold:.3f} "
              f"peak_candidates={len(peaks)}")

        # ── Step 6: Verify each peak has 2+ colours in raw data ───
        # Check RAW unsmoothed projections in a window around each peak
        # This is the ONLY filter — just confirm 2 colours are present
        # No other conditions. No minimum pixel counts beyond 1.

        bolt_hole_positions = []
        window = max(int(sigma * 3), 25)

        for peak_x in peaks:
            x0 = max(0, peak_x - window)
            x1 = min(len(purple_proj), peak_x + window)

            # Raw pixel count in window for each colour
            p_raw = purple_proj[x0:x1].sum()
            r_raw = red_proj[x0:x1].sum()
            g_raw = grey_proj[x0:x1].sum()

            # A colour is present if it has at least 1 raw pixel
            # in the window around this peak
            p_present = p_raw >= 1
            r_present = r_raw >= 1
            g_present = g_raw >= 1

            colours_present = sum([p_present, r_present, g_present])

            # THE ONLY RULE:
            # 2 or more colours present = valid bolt hole
            # 1 or fewer colours = reject
            if colours_present >= 2:
                # Find y centroid from all colour pixels in band
                x0b = max(0, peak_x - 20)
                x1b = min(roi_bgr.shape[1], peak_x + 20)
                band = (
                    purple_mask[:, x0b:x1b] |
                    red_mask[:,    x0b:x1b] |
                    grey_mask[:,   x0b:x1b]
                )
                if band.any():
                    ys, _ = np.where(band)
                    cy = int(ys.mean())
                else:
                    cy = roi_bgr.shape[0] // 2

                bolt_hole_positions.append((int(peak_x), cy))

                print(f"[DETECT] ACCEPTED x={peak_x} "
                      f"P={p_raw:.0f} R={r_raw:.0f} G={g_raw:.0f} "
                      f"colours={colours_present}")
            else:
                print(f"[DETECT] REJECTED x={peak_x} "
                      f"P={p_raw:.0f} R={r_raw:.0f} G={g_raw:.0f} "
                      f"colours={colours_present} — single colour only")

        # ── Step 7: Draw annotated ROI ────────────────────────────
        annotated_roi = roi_bgr.copy()

        # Highlight detected colour pixels so user can see them
        annotated_roi[purple_mask] = (180, 0, 180)   # magenta = purple
        annotated_roi[red_mask]    = (0, 0, 220)     # bright red = red
        annotated_roi[grey_mask]   = (180, 180, 0)   # cyan = grey

        # Draw circle at each confirmed hole
        for (cx, cy) in bolt_hole_positions:
            cv2.circle(annotated_roi, (cx, cy), 18, (0, 220, 0), 2)

        count = len(bolt_hole_positions)
        cv2.putText(
            annotated_roi,
            f"Holes: {count}",
            (5, 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        print(f"[DETECT] FINAL count={count} "
              f"accepted={len(bolt_hole_positions)} "
              f"rejected={len(peaks)-len(bolt_hole_positions)}")

        return {
            "bolt_hole_count":     count,
            "bolt_hole_positions": bolt_hole_positions,
            "purple_mask":         purple_mask,
            "red_mask":            red_mask,
            "grey_mask":           grey_mask,
            "annotated_roi":       annotated_roi,
            "combined_signal":     combined,
            "purple_proj":         ps,
            "red_proj":            rs,
            "grey_proj":           gs,
        }

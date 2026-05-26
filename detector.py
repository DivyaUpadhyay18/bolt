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
    
    def detect(self, roi_bgr, sigma=10, min_distance=18, prominence=0.2):
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
        
        combined = np.maximum.reduce([combo_pg, combo_pr, combo_rg, combo_prg])

        # ── Step 5: Zero out channels below minimum signal ────────
        # A channel must have a meaningful smoothed signal to be
        # considered present in the combination.
        # This prevents smoothing bleed (tiny values near 0) from
        # creating false two-colour signals.
        #
        # min_signal_level is the minimum smoothed projection value
        # for a channel to count as "present" in the combination.
        # Bleed from Gaussian smoothing is typically < 0.5
        # A real co-located signal is typically > 1.0
        min_signal_level = 1.0

        ps_clean = np.where(ps >= min_signal_level, ps, 0.0)
        rs_clean = np.where(rs >= min_signal_level, rs, 0.0)
        gs_clean = np.where(gs >= min_signal_level, gs, 0.0)

        print(f"[DETECT] RAW projections (smoothed): "
              f"P_sum={ps.sum():.0f} R_sum={rs.sum():.0f} G_sum={gs.sum():.0f}")
        print(f"[DETECT] CLEAN projections (threshold={min_signal_level}): "
              f"P_clean={ps_clean.sum():.0f} R_clean={rs_clean.sum():.0f} G_clean={gs_clean.sum():.0f}")

        # Rebuild combo signals using cleaned projections
        combo_pg  = np.minimum(ps_clean, gs_clean)
        combo_pr  = np.minimum(ps_clean, rs_clean)
        combo_rg  = np.minimum(rs_clean, gs_clean)
        combo_prg = np.minimum(np.minimum(ps_clean, rs_clean), gs_clean)

        combined = np.maximum.reduce([combo_pg, combo_pr,
                                      combo_rg, combo_prg])

        # ── Step 6: Find peaks ────────────────────────────────────
        if combined.max() == 0:
            print(f"[DETECT] combined=0 after cleaning — "
                  f"no two-colour overlap found "
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
                "purple_mask":         purple_mask,
                "red_mask":            red_mask,
                "grey_mask":           grey_mask,
            }

        threshold = combined.max() * 0.15
        peaks, _  = find_peaks(
            combined,
            distance=min_distance,
            height=threshold,
            prominence=prominence
        )

        print(f"[DETECT] after cleaning: "
              f"combined_max={combined.max():.3f} "
              f"threshold={threshold:.3f} "
              f"peak_candidates={len(peaks)}")

        # ── Step 7: Strict raw pixel two-colour verification ──────
        # For every candidate peak, check RAW unsmoothed pixel counts
        # in a window around the peak.
        #
        # STRICT RULE:
        #   Each colour that is claimed to be present must have
        #   at least min_raw_pixels in the window.
        #   min_raw_pixels is set high enough to reject bleed
        #   but low enough to accept faint real signals.
        #
        # A scattered single colour will have many raw pixels of
        # its colour but ZERO raw pixels of any other colour.
        # The raw check catches this even if smoothing bleed fooled
        # the combined signal.

        min_raw_pixels = 3   # must have at least 3 raw pixels to count as present
        window         = max(int(sigma * 2), 15)
        bolt_hole_positions = []

        for peak_x in peaks:
            x0 = max(0, peak_x - window)
            x1 = min(len(purple_proj), peak_x + window)

            p_raw = purple_proj[x0:x1].sum()
            r_raw = red_proj[x0:x1].sum()
            g_raw = grey_proj[x0:x1].sum()

            p_present = p_raw >= min_raw_pixels
            r_present = r_raw >= min_raw_pixels
            g_present = g_raw >= min_raw_pixels

            colours_present = sum([p_present, r_present, g_present])

            # Log every candidate clearly
            print(f"[DETECT] peak_x={peak_x} "
                  f"raw: P={p_raw:.0f} R={r_raw:.0f} G={g_raw:.0f} | "
                  f"present: P={p_present} R={r_present} G={g_present} | "
                  f"colours_present={colours_present}")

            # THE ONLY RULE — must have 2 or more colours
            if colours_present < 2:
                print(f"[DETECT] REJECTED x={peak_x} — "
                      f"only {colours_present} colour present "
                      f"(single colour is never a bolt hole)")
                continue

            # Valid combos check
            combo1_ok = p_present and g_present            # P+G
            combo2_ok = p_present and r_present and g_present  # P+R+G
            combo3_ok = r_present and g_present            # R+G
            combo4_ok = p_present and r_present            # P+R

            if not any([combo1_ok, combo2_ok, combo3_ok, combo4_ok]):
                print(f"[DETECT] REJECTED x={peak_x} — "
                      f"no valid colour combination")
                continue

            # Find y centroid
            x0b  = max(0, peak_x - 20)
            x1b  = min(roi_bgr.shape[1], peak_x + 20)
            band = (purple_mask[:, x0b:x1b] |
                    red_mask[:,    x0b:x1b] |
                    grey_mask[:,   x0b:x1b])

            cy = int(np.where(band)[0].mean()) if band.any() \
                 else roi_bgr.shape[0] // 2

            bolt_hole_positions.append((int(peak_x), cy))
            print(f"[DETECT] ACCEPTED x={peak_x} y={cy} — "
                  f"combo: P={p_present} R={r_present} G={g_present}")

        print(f"[DETECT] FINAL: candidates={len(peaks)} "
              f"accepted={len(bolt_hole_positions)} "
              f"rejected={len(peaks) - len(bolt_hole_positions)}")

        # ── Step 8: Draw annotated ROI ────────────────────────────
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

"""
Find the B-Scan detection zone using "B Scan" text as anchor.
Uses OCR to locate text, then restricts line detection to that region.
"""

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("[WARNING] pytesseract not available - OCR detection disabled")


def find_bscan_roi(frame):
    """
    Find B-Scan ROI using "B Scan" text for y_top and red line for y_bot.
    
    Strategy:
    1. Use OCR/pixel search to find "B Scan" text → y_top of panel
    2. Search for red line below text → y_bot of panel
    3. Use blue lines ONLY to measure x_left and x_right
    4. Extract full B-Scan region from top of panel to red line
    
    Returns:
        dict with roi, y_top, y_bot, x_left, x_right, found (bool)
    """
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("[ROI] ========== FINDING B-SCAN REGION ==========")
    
    # ========================================================================
    # STEP 1 — Find "B Scan" text to determine y_top
    # ========================================================================
    y_text_bottom = find_bscan_text_y(frame_rgb)
    
    if y_text_bottom is None:
        print("[ROI] FAIL: Could not find 'B Scan' text in frame")
        return _return_empty_roi()
    
    y_top = y_text_bottom + 2

    # Find how far down the first blue line is from y_top
    # and skip past it completely
    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W       = frame.shape[:2]
    x_end      = int(W * 0.55)

    # Scan downward from y_top to find the first blue line
    y_first_blue_from_top = None
    for dy in range(0, 40):
        scan_y = y_top + dy
        if scan_y >= H:
            break
        row    = frame_rgb[scan_y, 0:x_end]
        R_row  = row[:, 0].astype(int)
        G_row  = row[:, 1].astype(int)
        B_row  = row[:, 2].astype(int)
        blue_count = int(((R_row < 100) & (G_row < 100) & (B_row > 130)).sum())
        if blue_count >= max(int(x_end * 0.06), 15):
            y_first_blue_from_top = scan_y
            print(f"[ROI] first blue line found at y={scan_y} "
                  f"blue_count={blue_count}")
            break

    if y_first_blue_from_top is not None:
        # Find where the blue line ends (bottom of blue line group)
        y_after_blue = y_first_blue_from_top
        for dy in range(0, 15):
            scan_y = y_first_blue_from_top + dy
            if scan_y >= H:
                break
            row    = frame_rgb[scan_y, 0:x_end]
            R_row  = row[:, 0].astype(int)
            G_row  = row[:, 1].astype(int)
            B_row  = row[:, 2].astype(int)
            blue_count = int(
                ((R_row < 100) & (G_row < 100) & (B_row > 130)).sum()
            )
            if blue_count >= max(int(x_end * 0.06), 15):
                y_after_blue = scan_y
            else:
                break

        # Set y_top to just below the entire first blue line
        y_top = y_after_blue + 3
        print(f"[ROI] y_top adjusted to skip first blue line: "
              f"y_top={y_top}")

    print(f"[ROI] y_top (just below 'B Scan' text and blue line) = {y_top}")
    
    # ========================================================================
    # STEP 2 — Find red line to determine y_bot
    # ========================================================================
    y_red = find_red_line_y(frame_rgb, y_top)
    
    if y_red is None:
        print("[ROI] FAIL: Could not find red line below B Scan text")
        return _return_empty_roi()
    
    y_bot = y_red - 2
    print(f"[ROI] y_bot (just above red line) = {y_bot}")
    
    if (y_bot - y_top) < 10:
        print(f"[ROI] ERROR: detection zone too thin {y_bot - y_top}px")
        return _return_empty_roi()
    
    print(f"[ROI] ✓ Detection zone: y_top={y_top} to y_bot={y_bot} "
          f"(height={y_bot - y_top}px)")
    
    # ========================================================================
    # STEP 3 — Find blue lines for x measurement
    # ========================================================================
    blue_lines = find_blue_lines_between(frame_rgb, y_top, y_bot)
    print(f"[ROI] Blue lines found at y={blue_lines}")
    
    if len(blue_lines) == 0:
        # Fallback: use middle of zone for x measurement
        y_for_x = (y_top + y_bot) // 2
        print(f"[ROI] No blue lines found, using zone midpoint y={y_for_x}")
    else:
        y_for_x = blue_lines[0]  # Use first blue line
    
    # ========================================================================
    # STEP 4 — Find x_left and x_right from blue line extent
    # ========================================================================
    x_left, x_right = find_panel_x_extent(frame, y_for_x, y_for_x, y_red)
    
    if (x_right - x_left) < int(W * 0.10):
        print(f"[ROI] WARNING: panel width too narrow "
              f"({x_right - x_left}px), check line detection")
    
    # ========================================================================
    # STEP 5 — Crop ROI
    # ========================================================================
    roi = frame[y_top:y_bot, x_left:x_right]
    
    print(f"[ROI] ========== SUCCESS ==========")
    print(f"[ROI] FINAL: y_top={y_top} y_bot={y_bot} "
          f"x_left={x_left} x_right={x_right} "
          f"roi_shape={roi.shape}")
    
    return {
        "roi": roi,
        "y_top": y_top,
        "y_bot": y_bot,
        "x_left": x_left,
        "x_right": x_right,
        "found": True,
    }


def find_bscan_text_y(frame_rgb):
    """
    Find the y-coordinate of the bottom of the "B Scan" text label.
    The B-Scan panel starts immediately below this y.

    Strategy:
    The "B Scan" text label always appears on a row that has
    ALL of these properties simultaneously:
      1. Very few dark pixels (just 2-3 words of text) — 
         between 3 and 60 dark pixels in the left 50% of frame
      2. Background is LIGHT GREY (not the dark dotted A-Scan background)
      3. Immediately below this row (within 25px) is a BLUE horizontal line
         OR a dark panel border line spanning at least 50px

    The A-Scan area above has:
      - Grid dot rows with many evenly spaced dark pixels > 60
      - Dark background regions
      - No blue line immediately below text rows

    This combination of conditions uniquely identifies the "B Scan" label row.
    """
    H, W = frame_rgb.shape[:2]

    # Search only in bottom 70% of frame height
    # "B Scan" label is never in the top 30%
    y_search_start = int(H * 0.30)
    y_search_end   = int(H * 0.85)
    x_search_end   = int(W * 0.50)

    # ── Method A: pytesseract OCR ──────────────────────────────
    try:
        if PYTESSERACT_AVAILABLE:
            search_region = frame_rgb[y_search_start:y_search_end,
                                  0:x_search_end]

            data = pytesseract.image_to_data(
                search_region,
                output_type=Output.DICT,
                config="--psm 11 --oem 3"
            )

            matches = []
            for i, text in enumerate(data["text"]):
                cleaned = text.strip().lower().replace(" ", "")
                if cleaned in ["bscan", "b-scan", "bsca", "scan"] and len(cleaned) >= 4:
                    conf = int(data["conf"][i])
                    if conf < 10:
                        continue
                    y_bottom = y_search_start + data["top"][i] + data["height"][i]
                    matches.append((conf, y_bottom))
                    print(f"[ANCHOR] OCR: '{text}' conf={conf} "
                          f"y_bottom={y_bottom}")

            if matches:
                # Take the highest confidence match
                matches.sort(reverse=True)
                y_text_bottom = matches[0][1]
                print(f"[ANCHOR] OCR selected y_text_bottom={y_text_bottom}")
                return y_text_bottom

            print("[ANCHOR] OCR did not find 'B Scan' text")

    except Exception as e:
        print(f"[ANCHOR] OCR failed: {e}")

    # ── Method B: Find "B Scan" row by three-condition check ──
    # Condition 1: Row has few dark pixels (text only, not grid)
    # Condition 2: Row background is light grey
    # Condition 3: A blue or dark border line exists below within 25px

    R_full = frame_rgb[y_search_start:y_search_end,
                       0:x_search_end, 0].astype(int)
    G_full = frame_rgb[y_search_start:y_search_end,
                       0:x_search_end, 1].astype(int)
    B_full = frame_rgb[y_search_start:y_search_end,
                       0:x_search_end, 2].astype(int)

    # Dark pixel mask (text pixels)
    dark_mask    = (R_full < 120) & (G_full < 120) & (B_full < 120)
    dark_per_row = dark_mask.sum(axis=1)

    # Light grey background mask
    # Light grey: all channels 170-245, low saturation
    grey_mask    = ((R_full > 170) & (R_full < 245) &
                    (np.abs(R_full - G_full) < 20) &
                    (np.abs(G_full - B_full) < 20))
    grey_per_row = grey_mask.sum(axis=1)

    # Blue pixel mask for border detection
    blue_mask    = (R_full < 100) & (G_full < 100) & (B_full > 130)
    blue_per_row = blue_mask.sum(axis=1)

    n_rows    = R_full.shape[0]
    min_blue  = max(int(x_search_end * 0.06), 15)
    min_grey  = max(int(x_search_end * 0.40), 50)

    print(f"[ANCHOR] Scanning rows {y_search_start} to "
          f"{y_search_end} for B Scan label...")

    for rel_y in range(n_rows - 25):
        abs_y = y_search_start + rel_y

        # Condition 1: Few dark pixels — text row only
        # Must have between 3 and 80 dark pixels
        # A-Scan grid rows have many more dark pixels
        if not (3 <= dark_per_row[rel_y] <= 80):
            continue

        # Condition 2: Mostly light grey background
        # At least 40% of the row must be grey
        if grey_per_row[rel_y] < min_grey:
            continue

        # Condition 3: Blue border line within 5 to 25 rows below
        blue_below = False
        for dy in range(5, 26):
            if rel_y + dy >= n_rows:
                break
            if blue_per_row[rel_y + dy] >= min_blue:
                blue_below = True
                print(f"[ANCHOR] Method B: text-like row at "
                      f"abs_y={abs_y} dark={dark_per_row[rel_y]} "
                      f"grey={grey_per_row[rel_y]} "
                      f"blue_border at +{dy}px")
                break

        if blue_below:
            # This row matches all three conditions
            # Return the y just below this text row
            y_text_bottom = abs_y + 3
            print(f"[ANCHOR] Method B found B Scan text at "
                  f"y_text_bottom={y_text_bottom}")
            return y_text_bottom

    # ── Method C: Last resort — find the text row with blue line ─
    # The line immediately above the B-Scan panel border also has
    # few dark pixels and light background. The B-Scan panel border
    # appears within 10px below it.

    print("[ANCHOR] Method B failed — trying Method C (fallback)")

    for rel_y in range(n_rows - 10):
        abs_y = y_search_start + rel_y

        # Look for a row with few dark pixels AND grey background
        if not (2 <= dark_per_row[rel_y] <= 100):
            continue
        if grey_per_row[rel_y] < min_grey:
            continue

        # Check if a blue line is within 3 to 10 rows below
        for dy in range(3, 11):
            if rel_y + dy >= n_rows:
                break
            if blue_per_row[rel_y + dy] >= min_blue:
                y_text_bottom = abs_y + 2
                print(f"[ANCHOR] Method C: found label row at "
                      f"abs_y={abs_y} → y_text_bottom={y_text_bottom}")
                return y_text_bottom

    print("[ANCHOR] All methods failed to find B Scan text anchor")
    return None


def find_red_line_y(frame_rgb, y_search_start):
    """
    Find the red TR baseline below the B-Scan text label.
    Returns y coordinate of red line.
    Returns None if not found.
    """
    H, W = frame_rgb.shape[:2]
    x_end = int(W * 0.65)
    
    R = frame_rgb[y_search_start:H, 0:x_end, 0].astype(int)
    G = frame_rgb[y_search_start:H, 0:x_end, 1].astype(int)
    B = frame_rgb[y_search_start:H, 0:x_end, 2].astype(int)
    
    red_mask = (R > 150) & (G < 80) & (B < 80)
    red_per_row = red_mask.sum(axis=1)
    
    min_span = max(int(x_end * 0.08), 20)
    
    # Find rows with enough red pixels
    red_rows = [y_search_start + y
                for y in range(len(red_per_row))
                if red_per_row[y] >= min_span]
    
    if not red_rows:
        # Retry with lower threshold
        min_span_retry = max(int(x_end * 0.03), 10)
        red_rows = [y_search_start + y
                    for y in range(len(red_per_row))
                    if red_per_row[y] >= min_span_retry]
    
    if not red_rows:
        print("[ROI] red line not found")
        return None
    
    # Group consecutive red rows
    groups = []
    current = [red_rows[0]]
    for r in red_rows[1:]:
        if r - current[-1] <= 10:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    
    y_red = int(np.median(groups[0]))
    print(f"[ROI] red line found at y={y_red}")
    return y_red


def find_blue_lines_between(frame_rgb, y_top, y_bot):
    """
    Find horizontal blue lines between y_top and y_bot.
    Returns list of y coordinates of blue lines found.
    """
    H, W = frame_rgb.shape[:2]
    x_end = int(W * 0.65)
    
    if y_bot <= y_top:
        return []
    
    search = frame_rgb[y_top:y_bot, 0:x_end]
    R = search[:, :, 0].astype(int)
    G = search[:, :, 1].astype(int)
    B = search[:, :, 2].astype(int)
    
    blue_mask = (R < 100) & (G < 100) & (B > 130)
    blue_per_row = blue_mask.sum(axis=1)
    min_span = max(int(x_end * 0.06), 15)
    
    blue_rows = [y_top + y
                 for y in range(search.shape[0])
                 if blue_per_row[y] >= min_span]
    
    if not blue_rows:
        return []
    
    # Group consecutive blue rows
    groups = []
    current = [blue_rows[0]]
    for r in blue_rows[1:]:
        if r - current[-1] <= 10:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    
    return [int(np.median(g)) for g in groups]


def find_panel_x_extent(frame, y_first_blue, y_second_blue, y_red):
    """
    Find x_left and x_right of the B-Scan content area by
    measuring the horizontal extent of the blue and red lines.

    The blue lines and red line span exactly the full width
    of the B-Scan content area from left edge to right edge.
    This works for any zoom level or resolution.

    Args:
        frame: BGR frame
        y_first_blue: y-coordinate of first blue line
        y_second_blue: y-coordinate of second blue line
        y_red: y-coordinate of red line

    Returns:
        (x_left, x_right) tuple
    """
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_line_x_extent(frame_rgb, y_row, color):
        """
        For a given row y_row, find leftmost and rightmost x
        where that colour appears.
        color = 'blue' or 'red'
        """
        if not (0 <= y_row < frame_rgb.shape[0]):
            return None, None

        row = frame_rgb[y_row, :, :]
        R = row[:, 0].astype(int)
        G = row[:, 1].astype(int)
        B = row[:, 2].astype(int)

        if color == 'blue':
            mask = (R < 100) & (G < 100) & (B > 130)
        else:  # red
            mask = (R > 150) & (G < 80) & (B < 80)

        xs = np.where(mask)[0]
        if len(xs) < 10:
            return None, None
        return int(xs.min()), int(xs.max())

    # Sample multiple rows near each line to get stable readings
    # Use ±3 rows around each detected line y-coordinate
    all_x_lefts = []
    all_x_rights = []

    for y_line, color in [
        (y_first_blue, 'blue'),
        (y_second_blue, 'blue'),
        (y_red, 'red')
    ]:
        for dy in [-3, -2, -1, 0, 1, 2, 3]:
            y_sample = y_line + dy
            if 0 <= y_sample < H:
                xl, xr = get_line_x_extent(frame_rgb, y_sample, color)
                if xl is not None and xr is not None:
                    # Only accept if the line spans at least 5% of frame width
                    if (xr - xl) > int(W * 0.05):
                        all_x_lefts.append(xl)
                        all_x_rights.append(xr)

    if not all_x_lefts:
        print("[ROI] x_extent: could not measure line extents")
        # Cannot determine — return full left portion as fallback
        return 0, int(W * 0.60)

    # Take median to ignore outliers
    x_left = int(np.median(all_x_lefts))
    x_right = int(np.median(all_x_rights))

    # Add small buffer on each side
    x_left = max(x_left - 2, 0)
    x_right = min(x_right + 2, W - 1)

    print(f"[ROI] x_left={x_left} x_right={x_right} "
          f"panel_width={x_right - x_left}px "
          f"({(x_right - x_left) / W * 100:.1f}% of frame width)")

    return x_left, x_right


def _return_empty_roi():
    """Return a minimal invalid ROI."""
    empty = np.zeros((1, 1, 3), dtype=np.uint8)
    return {
        "roi": empty,
        "y_top": 0,
        "y_bot": 1,
        "x_left": 0,
        "x_right": 1,
        "found": False,
    }


def draw_roi_overlay(frame, roi_dict):
    """
    Draw green rectangle around ROI on full frame.
    
    Args:
        frame: BGR frame
        roi_dict: dict from find_bscan_roi()
    
    Returns:
        Annotated frame
    """
    if not roi_dict["found"]:
        return frame.copy()
    
    ann_frame = frame.copy()
    
    x_left = roi_dict["x_left"]
    x_right = roi_dict["x_right"]
    y_top = roi_dict["y_top"]
    y_bot = roi_dict["y_bot"]
    
    cv2.rectangle(ann_frame, (x_left, y_top), (x_right, y_bot),
                  (0, 255, 0), 3)
    
    label = "B-SCAN ROI"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    
    text_x = x_left
    text_y = y_top - 5
    
    cv2.rectangle(ann_frame, (text_x - 2, text_y - th - 2),
                  (text_x + tw + 2, text_y + 2),
                  (0, 0, 0), -1)
    
    cv2.putText(ann_frame, label, (text_x, text_y),
                font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    return ann_frame

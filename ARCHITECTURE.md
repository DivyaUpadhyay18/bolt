# 🏗️ System Architecture — Bolt Hole Detection

Comprehensive architecture documentation with component relationships, data flows, and design patterns.

---

## 📐 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   User Interface Layer                           │
│                    (Streamlit app.py)                            │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Video Upload     │  │ Parameter Tuning │  │ Display      │  │
│  │ • File picker    │  │ • Sliders        │  │ • Frame view │  │
│  │ • 5GB limit      │  │ • Dropdowns      │  │ • Graphs     │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │Speed Optimizer   │  │ Debug Tools      │  │Export CSV    │  │
│  │ • Frame skip     │  │ • Mask overlay   │  │ • Results    │  │
│  │ • Resolution     │  │ • Projection     │  │ • Metrics    │  │
│  │ • ROI cache      │  │   graphs         │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└──────────────┬───────────────────────────────────────────────────┘
               │
               │ Video frames, parameters
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Video Processing Engine (app.py)                    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Main Loop:                                               │   │
│  │  for frame in video:                                     │   │
│  │    • Skip frames (if frame_skip > 1)                    │   │
│  │    • Find ROI (cached every N frames)                   │   │
│  │    • Scale resolution (if optimization enabled)         │   │
│  │    • Call detector.detect(roi)                          │   │
│  │    • Track holes with tracker.track()                   │   │
│  │    • Render frame (if not headless)                     │   │
│  │    • Measure FPS and metrics                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└────┬──────────────────────┬──────────────────────┬───────────────┘
     │                      │                      │
     │ roi               frame                  parameters
     │                      │
     ▼                      ▼                      ▼
┌─────────────────────┐  ┌──────────────────┐  ┌──────────────┐
│ panel_finder.py     │  │ detector.py      │  │ tracker.py   │
│ ROI Detection       │  │ Bolt Detection   │  │ Numbering    │
│                     │  │                  │  │              │
│ • Find text anchor  │  │ • Colour masks   │  │ • History    │
│ • Locate red line   │  │ • Projections    │  │ • Matching   │
│ • Find blue lines   │  │ • Peak detection │  │ • Labels     │
│ • Extract bounds    │  │ • Validation     │  │ • Export     │
└──────────┬──────────┘  └────────┬─────────┘  └──────────────┘
           │                      │
           │                      │ colour_rules.py
           │                      │
           │            ┌─────────┴─────────┐
           │            │                   │
           │            ▼                   ▼
           │        ┌──────────────────┐  ┌──────────────┐
           │        │get_purple_mask() │  │ get_red_mask │
           │        │get_grey_mask()   │  │              │
           │        │Blue line filter  │  │              │
           │        └──────────────────┘  └──────────────┘
           │
           ▼
       holes: [{'x': 150, 'y': 466}, ...]
              numbered_holes: {'BH-1': {...}, ...}
              
              
┌─────────────────────────────────────────────────────────────────┐
│                  Utilities Layer (utils.py)                      │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │ Visualization│  │ Debug Graphs     │  │ Export & Stats     │ │
│  │              │  │                  │  │                    │ │
│  │• Draw circles│  │• Projections     │  │• CSV export        │ │
│  │• Draw labels │  │• Combined signal │  │• Metrics calc      │ │
│  │• Draw bounds │  │• Peak markers    │  │• FPS measurement   │ │
│  └──────────────┘  └──────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
VIDEO INPUT
    │
    ▼
┌─────────────────┐
│ Read Frame      │ ◄──── Video codec handling
│ (1920 × 1004)   │       cv2.VideoCapture()
└────────┬────────┘
         │
         ▼
    ┌────────────────────────┐
    │ Frame Skip Decision    │
    │ (if idx % skip != 0)   │ ◄──── Performance: Skip 1, 2, 5, 10...
    │ → skip processing      │
    └────────┬───────────────┘
             │
             ├─ YES → Save to buffer, continue next frame
             │
             ▼ NO
         ┌──────────────────────────┐
         │ ROI Redetection Decision │
         │ (if idx % cache != 0)    │ ◄──── Performance: Cache 30, 50, 100
         │ → use cached bounds      │
         └────────┬─────────────────┘
                  │
                  ├─ YES (use cache) → jump to Extract ROI
                  │
                  ▼ NO (redetect)
              ┌──────────────────────────┐
              │ Find ROI (panel_finder)  │
              │                          │
              │ 1. OCR text "B Scan"     │
              │ 2. Find red TR line      │
              │ 3. Find blue boundaries  │
              │ 4. Extract region        │
              └────────┬─────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │ Extract ROI           │
           │ (y_top:y_bot, x_left) │ ◄──── Size: 107 × 489
           └───────────┬───────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │ Resolution Scaling     │ ◄──── Optional: 100%, 75%, 50%, 33%
           │ (cv2.resize if needed) │
           └────────────┬───────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │ Extract Colour Masks (colour_rules)│
        │                                    │
        │ For each of 3 colours:             │
        │  1. Convert BGR → HSV              │
        │  2. Apply hue range filter         │
        │  3. Apply saturation threshold     │
        │  4. Exclude blue lines             │
        │  5. Create 2D binary mask          │
        │                                    │
        │ Result: 3 masks (107 × 489)        │
        └────────┬────────────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ▼           ▼           ▼
   Purple      Red         Grey
   Mask        Mask        Mask
     │           │           │
     └───────────┼───────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ Generate Projections       │
    │ (vertical sum per column)  │
    │                            │
    │ purple_proj = Σ(col)       │
    │ red_proj = Σ(col)          │
    │ grey_proj = Σ(col)         │
    │                            │
    │ Result: 3 vectors (489px)  │
    └───────────┬────────────────┘
                │
                ▼
    ┌────────────────────────────────┐
    │ Smooth Projections             │
    │ (Gaussian filter)              │
    │                                │
    │ ps = gauss_filter(purple_proj) │ ◄──── σ = 3-25
    │ rs = gauss_filter(red_proj)    │
    │ gs = gauss_filter(grey_proj)   │
    │                                │
    │ Result: 3 smoothed vectors     │
    └──────────┬─────────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ Combine Colour Signals     │
    │ (2-colour verification)    │
    │                            │
    │ combo_pg = min(ps, gs)     │
    │ combo_pr = min(ps, rs)     │
    │ combo_rg = min(rs, gs)     │
    │ combo_prg = min(ps,rs,gs)  │
    │                            │
    │ combined = max(combos)     │
    │                            │
    │ Result: 1 combined vector  │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │ Clean Signals                  │
    │ (remove smoothing noise)       │
    │                                │
    │ if purple_proj < 1.0 →         │
    │   purple_proj = 0              │
    │ (same for red, grey)           │
    │                                │
    │ Result: Thresholded combined   │
    └──────────┬─────────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │ Detect Peaks               │
    │ (find_peaks from SciPy)    │
    │                            │
    │ peaks = find_peaks(        │
    │   combined,                │
    │   distance=min_distance,   │ ◄──── 10-60 px
    │   prominence=prominence    │ ◄──── 0.05-1.0
    │ )                          │
    │                            │
    │ Result: X-coordinates      │
    └──────────┬─────────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │ Validate Detections            │
    │ (strict 2-colour verification) │
    │                                │
    │ for each peak:                 │
    │   count colours_present        │
    │   if colours_present < 2:      │
    │     reject peak                │
    │   if any_colour < 3 pixels:    │
    │     reject peak                │
    │                                │
    │ Result: Valid hole positions   │
    └──────────┬─────────────────────┘
               │
               ▼
       ┌──────────────────┐
       │ holes:           │
       │ [{'x': 150,      │
       │   'y': 466},     │
       │  {'x': 280,      │
       │   'y': 466}]     │
       └────────┬─────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Track Holes (tracker)   │
    │                         │
    │ 1. Match to history     │
    │ 2. Assign labels        │
    │ 3. Update persistence   │
    │ 4. Create new labels    │
    │                         │
    │ Result: numbered_holes  │
    └──────────┬──────────────┘
               │
               ▼
       ┌──────────────────────┐
       │ numbered_holes:      │
       │ {'BH-1': {...},      │
       │  'BH-2': {...}}      │
       └────────┬─────────────┘
                │
                ├─────────────────────────────┐
                │                             │
                ▼ (if not headless)          ▼ (always)
         ┌─────────────────┐        ┌──────────────────┐
         │ Render Frame    │        │ Collect Metrics  │
         │                 │        │                  │
         │ 1. Draw circles │        │ • frame_time     │
         │ 2. Draw labels  │        │ • holes detected │
         │ 3. Display      │        │ • signal values  │
         │ 4. Log to buffer│        │ • FPS            │
         │                 │        │                  │
         │ Display Update  │        │ Store in history │
         │ Every N frames  │        │                  │
         └────────┬────────┘        └──────────────────┘
                  │                         │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Final Output Layer   │
                    │                      │
                    │ • Annotated frames   │
                    │ • Detection metrics  │
                    │ • FPS counter        │
                    │ • CSV export option  │
                    │ • Debug graphs       │
                    └──────────────────────┘
```

---

## 🧬 Class & Function Structure

### Core Detection Pipeline

```python
class BoltHoleDetector:
    """Main detector orchestrating the entire pipeline."""
    
    def __init__(self, sigma=10, min_distance=18, prominence=0.2):
        self.sigma = sigma
        self.min_distance = min_distance
        self.prominence = prominence
    
    def detect(self, roi):
        """Main detection pipeline."""
        # Step 1: Extract colour masks
        purple_mask = self._get_purple_mask(roi)
        red_mask = self._get_red_mask(roi)
        grey_mask = self._get_grey_mask(roi)
        
        # Step 2: Generate projections
        ps = purple_mask.sum(axis=0)
        rs = red_mask.sum(axis=0)
        gs = grey_mask.sum(axis=0)
        
        # Step 3: Smooth projections
        ps = gaussian_filter1d(ps, sigma=self.sigma)
        rs = gaussian_filter1d(rs, sigma=self.sigma)
        gs = gaussian_filter1d(gs, sigma=self.sigma)
        
        # Step 4: Combine signals
        combined = self._combine_signals(ps, rs, gs)
        
        # Step 5: Clean signals
        ps, rs, gs = self._clean_signals(ps, rs, gs)
        
        # Step 6: Detect peaks
        peaks = self._find_peaks(combined)
        
        # Step 7: Validate detections
        holes = self._validate_holes(peaks, ps, rs, gs)
        
        return holes
    
    def _get_purple_mask(self, roi):
        """Extract purple colour mask."""
        # Implementation in colour_rules.py
        pass
    
    def _combine_signals(self, ps, rs, gs):
        """Combine multi-colour signals."""
        combo_pg = np.minimum(ps, gs)
        combo_pr = np.minimum(ps, rs)
        combo_rg = np.minimum(rs, gs)
        combo_prg = np.minimum(combo_pg, rs)
        return np.maximum.reduce([combo_pg, combo_pr, combo_rg, combo_prg])
    
    def _validate_holes(self, peaks, ps, rs, gs):
        """Apply strict 2-colour verification."""
        valid_holes = []
        for peak_x in peaks:
            colours_present = (ps[peak_x] > 0) + (rs[peak_x] > 0) + (gs[peak_x] > 0)
            if colours_present >= 2:  # Strict verification
                if ps[peak_x] >= 3 and rs[peak_x] >= 3 and gs[peak_x] >= 3:
                    valid_holes.append({'x': peak_x, 'y': self.roi_y})
        return valid_holes
```

### Tracker Component

```python
class Tracker:
    """Maintains persistent hole numbering across frames."""
    
    def __init__(self):
        self.hole_history = {}  # {'BH-1': {...}, ...}
    
    def track(self, holes, frame_idx):
        """Match detected holes to existing history."""
        numbered_holes = {}
        used_labels = set()
        
        for hole in holes:
            # Find closest existing hole
            best_label = self._find_best_match(hole)
            
            if best_label and best_label not in used_labels:
                # Update existing hole
                self.hole_history[best_label]['last_frame'] = frame_idx
                self.hole_history[best_label]['positions'].append(hole)
                numbered_holes[best_label] = hole
                used_labels.add(best_label)
            else:
                # Create new hole label
                new_label = f"BH-{len(self.hole_history) + 1}"
                self.hole_history[new_label] = {
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'positions': [hole]
                }
                numbered_holes[new_label] = hole
        
        return numbered_holes
    
    def _find_best_match(self, hole):
        """Find closest existing hole within threshold."""
        min_distance = 30  # pixels
        best_label = None
        best_dist = float('inf')
        
        for label, history in self.hole_history.items():
            if history['last_frame'] < current_frame - 10:
                continue  # Skip stale holes
            
            last_pos = history['positions'][-1]
            dist = np.sqrt((hole['x'] - last_pos['x'])**2 + 
                          (hole['y'] - last_pos['y'])**2)
            
            if dist < best_dist and dist < min_distance:
                best_dist = dist
                best_label = label
        
        return best_label
```

---

## 🎯 Algorithm Flowchart

```
START
  │
  ├─ Load video
  │
  ├─ For each frame:
  │
  ├─ Decision: Skip frame?
  │  ├─ YES → Go to next frame
  │  │
  │  ├─ NO ↓
  │  │
  │  ├─ Decision: Redetect ROI?
  │  │  ├─ YES → Run panel_finder
  │  │  │
  │  │  ├─ NO → Use cached bounds
  │  │  │
  │  │  ├─ ↓
  │  │
  │  ├─ Extract ROI from frame
  │  │
  │  ├─ Decision: Scale resolution?
  │  │  ├─ YES → cv2.resize()
  │  │  │
  │  │  ├─ NO → Keep original
  │  │  │
  │  │  ├─ ↓
  │  │
  │  ├─ For each colour (Purple, Red, Grey):
  │  │  ├─ Extract mask (colour_rules)
  │  │  ├─ Generate projection
  │  │  ├─ Apply Gaussian smoothing
  │  │  │
  │  │  └─ ↓
  │  │
  │  ├─ Combine colour signals (multi-colour verification)
  │  │
  │  ├─ Clean signals (remove noise)
  │  │
  │  ├─ Detect peaks in combined signal
  │  │
  │  ├─ For each peak:
  │  │  ├─ Count colours present
  │  │  │
  │  │  ├─ Decision: ≥ 2 colours?
  │  │  │  ├─ NO → Reject peak
  │  │  │  │
  │  │  │  ├─ YES ↓
  │  │  │
  │  │  ├─ Decision: ≥ 3 pixels/colour?
  │  │  │  ├─ NO → Reject peak
  │  │  │  │
  │  │  │  ├─ YES ↓
  │  │  │
  │  │  └─ Accept peak as valid hole
  │  │
  │  ├─ Track holes (assign labels BH-1, BH-2, ...)
  │  │
  │  ├─ Decision: Headless mode?
  │  │  ├─ NO:
  │  │  │  ├─ Render frame with annotations
  │  │  │  ├─ Draw circles, labels
  │  │  │  └─ Display (if display_update_freq check)
  │  │  │
  │  │  ├─ YES:
  │  │  │  └─ Skip rendering
  │  │  │
  │  │  ├─ ↓
  │  │
  │  ├─ Collect metrics (frame_time, FPS)
  │  │
  │  └─ ↓
  │
  ├─ Calculate final statistics
  ├─ Display results & performance
  ├─ Offer CSV export
  │
  END
```

---

## 🔌 Interface Contracts

### detector.py API

```python
# Input
roi: np.ndarray (shape: 107×489×3, dtype: uint8, format: BGR)
sigma: float (3-25)
min_distance: int (10-60)
prominence: float (0.05-1.0)

# Output
holes: List[Dict[str, Any]]
[
    {
        'x': int (0-489),
        'y': int (0-107),
        'signal': float (0-255)
    },
    ...
]

# Exceptions
ValueError: If parameters out of range
"""

### panel_finder.py API

```python
# Input
frame: np.ndarray (shape: H×W×3, dtype: uint8, format: BGR)

# Output
roi_bounds: Tuple[int, int, int, int]
(y_top, y_bot, x_left, x_right)
# Typical: (466, 573, 50, 539)

# Exceptions
ValueError: If ROI not found (returns default estimate)
"""

### tracker.py API

```python
# Input
holes: List[Dict[str, Any]]  # From detector.detect()
frame_idx: int

# Output
numbered_holes: Dict[str, Dict[str, Any]]
{
    'BH-1': {'x': 150, 'y': 466},
    'BH-2': {'x': 280, 'y': 466},
    ...
}

# State
hole_history: Dict[str, Dict]  # Persists across calls
"""

---

## 📊 State Diagram

```
App Lifecycle:

STARTUP
   │
   ├─ Load Streamlit session
   ├─ Initialize detector (parameters from sliders)
   ├─ Initialize tracker (empty history)
   ├─ Initialize ROI cache
   │
   ▼
WAITING_FOR_VIDEO
   │
   ├─ Display upload widget
   ├─ Wait for user file upload
   │
   ▼
VIDEO_LOADED
   │
   ├─ Parse video metadata
   ├─ Display frame count, duration, FPS
   ├─ Wait for "Start Processing" button
   │
   ▼
PROCESSING
   │
   ├─ Read frame
   ├─ Apply frame skip
   ├─ Apply ROI cache
   ├─ Run detection
   ├─ Run tracking
   ├─ Store results
   ├─ Render (if not headless)
   │
   ├─ Loop until video ends
   │
   ▼
PROCESSING_COMPLETE
   │
   ├─ Calculate final statistics
   ├─ Display results summary
   ├─ Show performance metrics
   ├─ Offer CSV download
   │
   ▼
READY_FOR_NEW_VIDEO
   │
   └─ Reset state, return to VIDEO_LOADED
```

---

## 🔍 Dependency Graph

```
External Libraries:
├─ OpenCV (cv2)
│  ├─ Image I/O (read/write)
│  ├─ Colour conversion (BGR↔HSV)
│  ├─ Image processing (resize, draw)
│  └─ Video capture
│
├─ NumPy
│  ├─ Array operations (sum, minimum, maximum)
│  ├─ Masking and indexing
│  ├─ Signal operations
│  └─ Statistics
│
├─ SciPy
│  ├─ gaussian_filter1d (smoothing)
│  ├─ find_peaks (peak detection)
│  └─ Signal processing
│
├─ Streamlit
│  ├─ Web UI
│  ├─ Session state
│  ├─ File upload
│  └─ Display rendering
│
├─ Matplotlib
│  ├─ Debug graphs (projections, peaks)
│  └─ Visualization
│
├─ Pandas
│  └─ CSV export
│
└─ Pytesseract
   └─ OCR text detection

Internal Modules:
├─ colour_rules.py
│  ├─ Used by: detector.py
│  └─ Uses: cv2, numpy
│
├─ panel_finder.py
│  ├─ Used by: app.py
│  ├─ Uses: cv2, pytesseract
│  └─ Optional: colour_rules (ROI validation)
│
├─ detector.py
│  ├─ Used by: app.py
│  ├─ Uses: numpy, scipy, colour_rules
│  └─ Imports: colour_rules
│
├─ tracker.py
│  ├─ Used by: app.py
│  ├─ Uses: numpy
│  └─ Imports: None
│
├─ utils.py
│  ├─ Used by: app.py
│  ├─ Uses: cv2, numpy, matplotlib, pandas
│  └─ Imports: None
│
└─ app.py
   ├─ Uses: streamlit, cv2, numpy, pandas, matplotlib
   └─ Imports: detector, tracker, panel_finder, colour_rules, utils
```

---

## ⚙️ Tuning Parameter Sensitivity

```
Detection Quality vs Performance Trade-offs:

┌────────────┬─────────┬──────────────┬──────────────┐
│ Parameter  │ Default │ Effect       │ Sensitivity  │
├────────────┼─────────┼──────────────┼──────────────┤
│ sigma      │ 10      │ ↓ → false -  │ High         │
│            │         │ ↑ → misses   │              │
├────────────┼─────────┼──────────────┼──────────────┤
│ min_dist   │ 18      │ ↓ → noise    │ High         │
│            │         │ ↑ → misses   │              │
├────────────┼─────────┼──────────────┼──────────────┤
│ prominence │ 0.20    │ ↓ → noise    │ Medium       │
│            │         │ ↑ → misses   │              │
├────────────┼─────────┼──────────────┼──────────────┤
│ frame_skip │ 2       │ → 2x speed   │ Low (FPS)    │
├────────────┼─────────┼──────────────┼──────────────┤
│ resolution │ 100%    │ 50% → 4x     │ Low (FPS)    │
├────────────┼─────────┼──────────────┼──────────────┤
│ roi_cache  │ 30      │ 100 → 1.3x   │ Low (FPS)    │
└────────────┴─────────┴──────────────┴──────────────┘

Recommended Tuning Order:
1. Start with defaults (sigma=10, distance=18, prominence=0.20)
2. If false positives: increase sigma or prominence
3. If false negatives: decrease sigma or prominence
4. For speed: increase frame_skip, reduce resolution
5. Fine-tune colours in colour_rules.py
```

---

## 🚀 Performance Benchmarks

```
Hardware: NVIDIA RTX 4050 (6GB VRAM)
Video: 1920×1004 @ 30 FPS

Configuration 1: Default (baseline)
├─ Frame skip: 1
├─ Resolution: 100%
├─ ROI cache: 30
├─ Display refresh: 1
├─ Headless: OFF
└─ Performance: 25 FPS

Configuration 2: Balanced
├─ Frame skip: 3
├─ Resolution: 75%
├─ ROI cache: 50
├─ Display refresh: 5
├─ Headless: OFF
└─ Performance: 120 FPS

Configuration 3: Batch/Export
├─ Frame skip: 10
├─ Resolution: 50%
├─ ROI cache: 100
├─ Display refresh: 30
├─ Headless: ON
└─ Performance: 600+ FPS
```

---

**Document Version:** 2.0  
**Last Updated:** 2026-05-26  
**Audience:** Developers, Contributors, Maintainers

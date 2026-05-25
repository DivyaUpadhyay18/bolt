# Quick Start Guide — Horizontal Projection Bolt Hole Detector

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd c:\Users\divya\Downloads\bolt
pip install -r requirements.txt
```

### Step 2: Choose Your Interface

#### **Option A: Web UI (Recommended for Exploration)**

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Features:**
- Upload ROI images
- Adjust parameters with sliders
- See results in real-time
- View debug graphs and mask visualizations

#### **Option B: Command Line (Quick One-Off Tests)**

```bash
python test_detector.py path/to/roi.png [sigma] [min_distance] [prominence]
```

Example:
```bash
python test_detector.py test_roi.png 8 20 0.3
```

**Output:** Saves results to `detection_output/`

#### **Option C: Python API (Integration)**

```python
from detector import BoltHoleDetector
import cv2

roi = cv2.imread("roi.png")
detector = BoltHoleDetector(sigma=8, min_distance=20, prominence=0.3)
result = detector.detect(roi)

print(f"Found {result['bolt_hole_count']} holes")
print(f"Positions: {result['bolt_hole_positions']}")
```

### Step 3: Optimize Parameters

If results aren't perfect, adjust in Streamlit sidebar or command line:

| Problem | Solution |
|---------|----------|
| Missing holes | Increase σ (8→12→15) or decrease prominence (0.3→0.2) |
| False positives | Decrease σ (8→5) or increase prominence (0.3→0.5) |
| Colours not detected | Check masks in debug panel, adjust HSV ranges in `colour_rules.py` |

---

## 📊 Understanding the Output

### Annotated ROI
- **Green circles** = Detected bolt holes
- **Green dot in center** = Precise (x, y) coordinate

### Projection Debug Graph
- **Purple line** = Purple pixel density per x-position
- **Grey line** = Grey pixel density per x-position
- **Green line** = Purple × Grey (combined signal)
- **Red dotted lines** = Detected peaks (holes)

A bolt hole = where **both purple AND grey peak simultaneously**.

### Colour Masks
- **Magenta = Purple pixels** detected in the image
- **Cyan = Grey pixels** detected in the image
- If a mask is empty, the colour is missing → adjust HSV ranges

---

## 🔧 Troubleshooting

### "No holes detected"

1. **Check masks:** Run with `--show-masks` or in Streamlit, look at mask visualizations
2. **Adjust σ:** Try higher values (10, 12, 15) to bridge larger gaps
3. **Loosen colours:** Edit `colour_rules.py` HSV ranges to capture fainter pixels
4. **Reduce prominence:** Lower threshold to 0.1-0.2

### "Too many false positives"

1. **Reduce σ:** Try smaller values (5, 6)
2. **Increase prominence:** Higher threshold (0.5, 1.0)
3. **Increase min_distance:** Prevent splitting single holes

### "Wrong colours detected"

1. Look at purple/grey masks in debug panel
2. If one is empty, HSV range is too strict
3. Edit `colour_rules.py` — widen lower/upper bounds
4. Re-run detector

---

## 📝 Example Workflow

### 1. Extract a test frame from your video

```python
import cv2

cap = cv2.VideoCapture("video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)  # Frame 1000
ret, frame = cap.read()
cap.release()

# Extract ROI between reference lines (example)
roi = frame[600:950, :]
cv2.imwrite("roi.png", roi)
```

### 2. Run detector with default parameters

```bash
python test_detector.py roi.png
```

Check `detection_output/` for results.

### 3. If unsatisfied, open Streamlit and tweak

```bash
streamlit run app.py
```

Upload the same `roi.png`, adjust sliders, observe real-time results.

### 4. Once happy, use Python API in production

```python
detector = BoltHoleDetector(sigma=YOUR_SIGMA, min_distance=YOUR_DIST, prominence=YOUR_PROM)
result = detector.detect(roi)
hole_count = result['bolt_hole_count']
positions = result['bolt_hole_positions']
```

---

## 📚 More Resources

- **Full documentation:** `README_HORIZONTAL_PROJECTION.md`
- **Tech details:** How horizontal projection works
- **Colour tuning:** Adjust HSV ranges for different lighting
- **Integration:** Use detector in video processing pipeline

---

## ✅ Test Results (Frame 1000)

```
Holes detected: 19
Positions:
  1. (53, 70)      12. (831, 68)
  2. (156, 131)    13. (904, 77)
  3. (191, 160)    14. (963, 71)
  4. (467, 44)     15. (1031, 77)
  5. (489, 44)     16. (1083, 76)
  6. (574, 71)     17. (1131, 66)
  7. (616, 73)     18. (1217, 121)
  8. (657, 73)     19. (1247, 110)
  9. (703, 75)
 10. (733, 75)
 11. (769, 77)
```

---

**Status:** ✅ Implementation complete and tested  
**Next steps:** Try Streamlit UI, adjust parameters for your specific images

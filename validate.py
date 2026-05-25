"""
Validation script — Check that all dependencies are installed correctly.
"""

import sys
from pathlib import Path

print("="*70)
print("BOLT HOLE DETECTOR — Dependency Validation")
print("="*70)

# Check Python version
print(f"\nPython: {sys.version.split()[0]}")
assert sys.version_info >= (3, 7), "Python 3.7+ required"
print("  ✅ OK")

# Check required packages
packages = {
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'PIL': 'Pillow',
    'streamlit': 'Streamlit',
}

print(f"\nPackages:")
missing = []
for module, name in packages.items():
    try:
        __import__(module)
        version = getattr(__import__(module), '__version__', 'unknown')
        print(f"  ✅ {name:<20} {version}")
    except ImportError:
        print(f"  ❌ {name:<20} MISSING")
        missing.append(module)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print(f"\nInstall with:")
    print(f"  pip install -r requirements.txt")
    sys.exit(1)

# Check source files
print(f"\nSource files:")
required_files = [
    'detector.py',
    'colour_rules.py',
    'utils.py',
    'app.py',
    'test_detector.py',
    'demo_video.py',
    'requirements.txt',
]

missing_files = []
for f in required_files:
    path = Path(f)
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"  ✅ {f:<30} ({size_kb:.1f} KB)")
    else:
        print(f"  ❌ {f:<30} MISSING")
        missing_files.append(f)

if missing_files:
    print(f"\n❌ Missing files: {', '.join(missing_files)}")
    sys.exit(1)

# Test detector import
print(f"\nDetector:")
try:
    from detector import BoltHoleDetector
    print(f"  ✅ BoltHoleDetector class imported successfully")
except Exception as e:
    print(f"  ❌ Error importing detector: {e}")
    sys.exit(1)

# Test colour_rules import
print(f"\nColour rules:")
try:
    from colour_rules import get_purple_mask, get_grey_mask
    print(f"  ✅ Colour functions imported successfully")
except Exception as e:
    print(f"  ❌ Error importing colour_rules: {e}")
    sys.exit(1)

# Test utils import
print(f"\nUtils:")
try:
    from utils import plot_projection_debug, draw_masks_overlay
    print(f"  ✅ Utility functions imported successfully")
except Exception as e:
    print(f"  ❌ Error importing utils: {e}")
    sys.exit(1)

print(f"\n" + "="*70)
print("✅ ALL CHECKS PASSED — System ready to use!")
print("="*70)

print(f"\nNext steps:")
print(f"  1. Web UI:  streamlit run app.py")
print(f"  2. CLI:     python test_detector.py roi.png")
print(f"  3. Video:   python demo_video.py")
print(f"  4. API:     from detector import BoltHoleDetector")

print(f"\n")

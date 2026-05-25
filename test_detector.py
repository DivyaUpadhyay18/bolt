"""
Standalone test script for bolt hole detector.
Runs detection on a single image and shows results.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from detector import BoltHoleDetector
from utils import plot_projection_debug, draw_masks_overlay


def test_detector_on_image(image_path, sigma=8, min_distance=20, prominence=0.3):
    """
    Test detector on a single image.
    
    Args:
        image_path: Path to image file
        sigma: Gaussian smoothing parameter
        min_distance: Minimum distance between holes
        prominence: Peak prominence threshold
    """
    
    # Load image
    print(f"Loading image: {image_path}")
    roi_bgr = cv2.imread(image_path)
    
    if roi_bgr is None:
        print(f"ERROR: Could not load image from {image_path}")
        return False
    
    print(f"Image size: {roi_bgr.shape}")
    
    # Create detector
    print(f"\nDetector parameters:")
    print(f"  σ (sigma) = {sigma}")
    print(f"  min_distance = {min_distance}")
    print(f"  prominence = {prominence}")
    
    detector = BoltHoleDetector(sigma=sigma, min_distance=min_distance, prominence=prominence)
    
    # Run detection
    print("\nRunning detection...")
    result = detector.detect(roi_bgr)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Holes detected: {result['bolt_hole_count']}")
    print(f"Positions (x, y):")
    for i, (x, y) in enumerate(result['bolt_hole_positions'], 1):
        print(f"  {i:2d}. ({x:4d}, {y:4d})")
    print(f"{'='*60}\n")
    
    # Save outputs
    output_dir = Path("detection_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save annotated ROI
    annotated_path = output_dir / "annotated_roi.png"
    cv2.imwrite(str(annotated_path), result['annotated_roi'])
    print(f"✓ Saved annotated ROI: {annotated_path}")
    
    # Save projection debug graph
    debug_path = output_dir / "projection_debug.png"
    cv2.imwrite(str(debug_path), plot_projection_debug(result))
    print(f"✓ Saved projection debug: {debug_path}")
    
    # Save mask overlay
    overlay_path = output_dir / "masks_overlay.png"
    overlay = draw_masks_overlay(roi_bgr, result)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"✓ Saved masks overlay: {overlay_path}")
    
    # Save individual masks
    purple_path = output_dir / "purple_mask.png"
    cv2.imwrite(str(purple_path), result['purple_mask'].astype(np.uint8) * 255)
    print(f"✓ Saved purple mask: {purple_path}")
    
    grey_path = output_dir / "grey_mask.png"
    cv2.imwrite(str(grey_path), result['grey_mask'].astype(np.uint8) * 255)
    print(f"✓ Saved grey mask: {grey_path}")
    
    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    
    return True


def extract_frame_from_video(video_path, frame_num=100):
    """
    Extract a single frame from video for testing.
    
    Args:
        video_path: Path to video file
        frame_num: Frame number to extract
    
    Returns:
        Frame as BGR image, or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"ERROR: Could not read frame {frame_num}")
        return None
    
    return frame


if __name__ == "__main__":
    print("="*60)
    print("BOLT HOLE DETECTOR — Test Script")
    print("="*60)
    
    # Check for test image
    test_images = [
        Path("test_roi.png"),
        Path("sample.png"),
        Path("roi.png"),
        Path("test.png"),
    ]
    
    image_path = None
    for path in test_images:
        if path.exists():
            image_path = path
            break
    
    if image_path is None:
        print("\nNo test image found.")
        print("Tried to find:")
        for path in test_images:
            print(f"  - {path}")
        
        print("\nUsage:")
        print("  python test_detector.py <image_path> [sigma] [min_distance] [prominence]")
        print("\nExample:")
        print("  python test_detector.py roi.png 8 20 0.3")
        sys.exit(1)
    
    # Parse command line args
    kwargs = {}
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['sigma'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kwargs['min_distance'] = int(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs['prominence'] = float(sys.argv[4])
    
    # Run test
    success = test_detector_on_image(image_path, **kwargs)
    
    if not success:
        sys.exit(1)

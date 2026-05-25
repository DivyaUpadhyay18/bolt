"""
Utility functions for visualization and frame processing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_projection_debug(result):
    """
    Plot projection signals with peak markers.
    
    Args:
        result: dict from detector.detect()
    
    Returns:
        BGR image of plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    
    purple_proj = result.get("purple_proj", np.array([]))
    red_proj = result.get("red_proj", np.array([]))
    grey_proj = result.get("grey_proj", np.array([]))
    combined = result.get("combined_signal", np.array([]))
    positions = result.get("bolt_hole_positions", [])
    
    # Extract x-coordinates of positions
    peak_xs = [p[0] for p in positions] if positions else []
    
    # Purple signal
    axes[0].plot(purple_proj, color='purple', linewidth=1.5)
    for px in peak_xs:
        axes[0].axvline(px, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title("Purple (37F) Signal", fontsize=10)
    axes[0].set_ylabel("Pixel count")
    axes[0].grid(alpha=0.3)
    
    # Red signal
    axes[1].plot(red_proj, color='red', linewidth=1.5)
    for px in peak_xs:
        axes[1].axvline(px, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title("Red (TR) Signal", fontsize=10)
    axes[1].set_ylabel("Pixel count")
    axes[1].grid(alpha=0.3)
    
    # Grey signal
    axes[2].plot(grey_proj, color='grey', linewidth=1.5)
    for px in peak_xs:
        axes[2].axvline(px, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title("Grey (37R) Signal", fontsize=10)
    axes[2].set_ylabel("Pixel count")
    axes[2].grid(alpha=0.3)
    
    # Combined signal
    axes[3].plot(combined, color='black', linewidth=2)
    for px in peak_xs:
        axes[3].axvline(px, color='red', linestyle='--', linewidth=2, label='Detected' if px == peak_xs[0] else '')
    axes[3].set_title("Combined Signal", fontsize=10)
    axes[3].set_ylabel("Pixel count")
    axes[3].set_xlabel("X position (pixels)")
    axes[3].grid(alpha=0.3)
    if peak_xs:
        axes[3].legend()
    
    plt.tight_layout()
    
    # Convert plot to image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return bgr_img


def draw_masks_overlay(roi_bgr, result):
    """
    Draw colour masks as overlay on ROI.
    
    Args:
        roi_bgr: BGR ROI
        result: dict from detector.detect()
    
    Returns:
        BGR image with mask overlay
    """
    overlay = roi_bgr.copy()
    
    purple_mask = result.get("purple_mask")
    red_mask = result.get("red_mask")
    grey_mask = result.get("grey_mask")
    
    alpha = 0.5
    
    if purple_mask is not None:
        overlay[purple_mask] = cv2.addWeighted(
            roi_bgr[purple_mask], 1-alpha, np.uint8([180, 0, 180]), alpha, 0
        )
    
    if red_mask is not None:
        overlay[red_mask] = cv2.addWeighted(
            roi_bgr[red_mask], 1-alpha, np.uint8([0, 0, 220]), alpha, 0
        )
    
    if grey_mask is not None:
        overlay[grey_mask] = cv2.addWeighted(
            roi_bgr[grey_mask], 1-alpha, np.uint8([180, 180, 0]), alpha, 0
        )
    
    return overlay


def draw_numbered_holes(roi, active_holes):
    """
    Draw numbered green circles with labels on ROI.
    
    Args:
        roi: BGR ROI
        active_holes: list of hole dicts from tracker.update()
    
    Returns:
        Annotated ROI
    """
    ann_roi = roi.copy()
    
    for hole in active_holes:
        cx, cy = hole["cx"], hole["cy"]
        label = hole["label"]
        
        # Draw filled green circle
        cv2.circle(ann_roi, (cx, cy), 20, (0, 180, 0), -1)
        
        # Draw white border
        cv2.circle(ann_roi, (cx, cy), 20, (255, 255, 255), 2)
        
        # Draw label centered inside
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.42
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        
        tx = cx - tw // 2
        ty = cy + th // 2
        
        cv2.putText(ann_roi, label, (tx, ty),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return ann_roi

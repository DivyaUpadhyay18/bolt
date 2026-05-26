"""
Colour mask extraction for bolt hole detection.
Purple (37F), Red (TR), and Grey (37R) channels.
"""

import numpy as np


def get_purple_mask(roi_rgb):
    """
    Extract purple (37F forward probe) mask from RGB ROI.
    
    Args:
        roi_rgb: uint8 array (H, W, 3) in RGB order
    
    Returns:
        bool ndarray (H, W) where True = purple pixel
    """
    r = roi_rgb[:, :, 0].astype(int)
    g = roi_rgb[:, :, 1].astype(int)
    b = roi_rgb[:, :, 2].astype(int)
    
    # Exclude blue line pixels first
    blue_line = (r < 100) & (g < 100) & (b > 130)
    
    purple_mask = (
        (r >= 85) & (r <= 215) &
        (g >= 25) & (g <= 140) &
        (b >= 85) & (b <= 215) &
        (np.abs(r - b) < 90) &
        ((r - g) > 12) &
        ((b - g) > 12) &
        ~blue_line
    )
    
    return purple_mask


def get_red_mask(roi_rgb):
    """
    Extract red (TR through-transmission) mask from RGB ROI.
    
    Args:
        roi_rgb: uint8 array (H, W, 3) in RGB order
    
    Returns:
        bool ndarray (H, W) where True = red pixel
    """
    r = roi_rgb[:, :, 0].astype(int)
    g = roi_rgb[:, :, 1].astype(int)
    b = roi_rgb[:, :, 2].astype(int)
    
    # Exclude blue line pixels
    blue_line = (r < 100) & (g < 100) & (b > 130)
    
    red_mask = (
        (r >= 140) & (r <= 255) &
        (g >= 0) & (g <= 90) &
        (b >= 0) & (b <= 90) &
        ((r - g) > 80) &
        ((r - b) > 80) &
        ~blue_line
    )
    
    return red_mask


def get_grey_mask(roi_rgb):
    """
    Extract grey (37R reverse probe) mask from RGB ROI.
    
    Args:
        roi_rgb: uint8 array (H, W, 3) in RGB order
    
    Returns:
        bool ndarray (H, W) where True = grey pixel
    """
    r = roi_rgb[:, :, 0].astype(int)
    g = roi_rgb[:, :, 1].astype(int)
    b = roi_rgb[:, :, 2].astype(int)
    
    # Exclude blue line pixels
    blue_line = (r < 100) & (g < 100) & (b > 130)
    
    grey_mask = (
        (np.abs(r - g) < 28) &
        (np.abs(g - b) < 28) &
        (np.abs(r - b) < 28) &
        (r >= 45) & (r <= 195) &
        ~blue_line
    )
    
    return grey_mask

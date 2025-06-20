import numpy as np


def bin_image(img, factor=2):
    """Bin the image by the given factor."""
    if factor <= 1:
        return img
    h, w = img.shape
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    img = img[:h_trim, :w_trim]
    img = img.reshape(h_trim // factor, factor, w_trim // factor, factor)
    return img.mean(axis=(1, 3))

def normalize_image(img, clip_percent=1):
    """Stretch contrast by clipping low/high extremes (percentile-based)."""
    img = img.astype(np.float32)
    low, high = np.percentile(img, [clip_percent, 100 - clip_percent])
    img = np.clip(img, low, high)
    img -= img.min()
    img /= (img.max() + 1e-9)
    return (img * 255).astype(np.uint8)
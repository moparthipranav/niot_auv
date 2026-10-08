"""
Detect a white bin and classify red tape markers as X or O.

Uses classical computer vision (OpenCV + NumPy), without deep learning.
"""
import math
import sys

import cv2

import numpy as np


# ---------------------------------------------------------------- bin ----
def detect_bin(img):
    """Return (contour, mask) of the white bin, or (None, None)."""
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # White/grey plastic: low saturation, bright. Threshold V adaptively
    # using Otsu on low-saturation pixels to handle lighting changes.
    s, v = hsv[..., 1], hsv[..., 2]
    low_sat = s < 70
    if low_sat.sum() < 100:
        return None, None
    t, _ = cv2.threshold(v[low_sat].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (
        ((s < 70) & (v > max(t, 120))).astype(np.uint8) * 255
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    image_area = img.shape[0] * img.shape[1]
    if cv2.contourArea(c) < 0.03 * image_area:
        return None, None

    hull = cv2.convexHull(c)            # fills notches caused by the red tape
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [hull], -1, 255, -1)
    return hull, filled


def estimate_bin_pose(img, real_bin_width, fx):
    """Return normalized image offsets, estimated range, and contour."""
    if not math.isfinite(real_bin_width) or real_bin_width <= 0:
        return None
    if not math.isfinite(fx) or fx <= 0:
        return None

    bin_hull, _ = detect_bin(img)
    if bin_hull is None:
        return None

    x, y, width_pixels, height_pixels = cv2.boundingRect(bin_hull)
    if width_pixels <= 0:
        return None

    image_height, image_width = img.shape[:2]
    center_x = x + width_pixels / 2.0
    center_y = y + height_pixels / 2.0
    norm_x = (center_x - image_width / 2.0) / (image_width / 2.0)
    norm_y = (center_y - image_height / 2.0) / (image_height / 2.0)
    range_m = fx * real_bin_width / width_pixels
    return norm_x, norm_y, range_m, bin_hull


def estimate_bin_detection(img, real_bin_width, fx):
    """Return the bin and marker measurements used by the mission."""
    bin_hull, bin_mask = detect_bin(img)
    if bin_hull is None:
        return None

    image_height, image_width = img.shape[:2]
    x, y, width_pixels, height_pixels = cv2.boundingRect(bin_hull)
    center_x = x + width_pixels / 2.0
    center_y = y + height_pixels / 2.0
    norm_x = (center_x - image_width / 2.0) / (image_width / 2.0)
    norm_y = (center_y - image_height / 2.0) / (image_height / 2.0)

    range_m = 0.0
    if (math.isfinite(real_bin_width) and real_bin_width > 0
            and math.isfinite(fx) and fx > 0 and width_pixels > 0):
        range_m = fx * real_bin_width / width_pixels

    marker_mask = red_mask(img, bin_mask)
    label, info, marker_hull = classify_marker(marker_mask)
    yaw = math.radians(cv2.minAreaRect(bin_hull)[-1])
    confidence = 0.0
    if marker_hull is not None:
        moments = cv2.moments(marker_hull)
        if moments['m00'] > 0:
            center_x = moments['m10'] / moments['m00']
            center_y = moments['m01'] / moments['m00']
            norm_x = (center_x - image_width / 2.0) / (image_width / 2.0)
            norm_y = (center_y - image_height / 2.0) / (image_height / 2.0)
            confidence = info['confidence']

    return {
        'detected': True,
        'marker_detected': label in ('X', 'O'),
        'x': norm_x,
        'y': norm_y,
        'z': range_m,
        'yaw': yaw,
        'confidence': confidence,
        'bin_hull': bin_hull,
        'marker_hull': marker_hull,
    }


# ------------------------------------------------------------- marker ----
def red_mask(img, bin_mask):
    """Binary mask of the red/pink tape, only inside the bin."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Red = high 'a' channel in Lab (robust even when colours are washed out)
    a = lab[..., 1].astype(np.float32)
    inside = bin_mask > 0
    a_in = a[inside]
    # threshold relative to the bin's own colour distribution
    thr = max(np.percentile(a_in, 50) + 12, 140)
    m = (a > thr).astype(np.uint8)

    h = hsv[..., 0]
    hue_ok = (h < 15) | (h > 160)
    m = (m & hue_ok.astype(np.uint8)) * 255

    # Stay away from bin edges (orange zip-ties, rope sit on the rim)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    safe = cv2.erode(bin_mask, kernel)
    m = cv2.bitwise_and(m, safe)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def classify_marker(mask):
    """
    Classify the tape as X or O from a binary mask.

    Rotation / scale invariant. Returns (label, details dict, contour).
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 'none', {}, None

    # Merge all sizeable blobs (tape can be broken into pieces) into one shape
    min_area = 0.002 * mask.size
    big = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not big:
        return 'none', {}, None
    pts = np.vstack(big)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)

    shape = np.zeros_like(mask)
    cv2.drawContours(shape, [hull], -1, 255, -1)
    marker = cv2.bitwise_and(mask, shape)
    ink = cv2.countNonZero(marker)

    # --- Feature 1: is there a hole (ring)? -> O
    cnts2, hier2 = cv2.findContours(
        marker, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    hole_area = 0
    if hier2 is not None:
        for i, c in enumerate(cnts2):
            if hier2[0][i][3] != -1:
                hole_area = max(hole_area, cv2.contourArea(c))
    hole_ratio = hole_area / hull_area if hull_area else 0

    # --- Feature 2: is the centre of the hull painted? X: yes, O: no
    M = cv2.moments(hull)
    if M['m00'] <= 0 or hull_area <= 0:
        return 'none', {}, None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    r = max(3, int(0.06 * np.sqrt(hull_area)))
    patch = marker[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
    centre_fill = patch.mean() / 255 if patch.size else 0

    # --- Feature 3: fill ratio (ink / hull). X ~0.2-0.4, ring ~0.4-0.6
    fill = ink / hull_area if hull_area else 0

    # Voting
    o_votes = 0
    o_votes += hole_ratio > 0.08
    o_votes += centre_fill < 0.25
    o_votes += fill > 0.45 and centre_fill < 0.5
    label = 'O' if o_votes >= 2 else 'X'

    return label, {
        'hole_ratio': round(hole_ratio, 3),
        'centre_fill': round(centre_fill, 3),
        'fill': round(fill, 3),
        'votes_for_O': int(o_votes),
        'confidence': round(max(o_votes, 3 - o_votes) / 3.0, 3),
    }, hull


# --------------------------------------------------------------- main ----
def run(path, debug=False):
    """Run bin and marker detection on one image and save the overlay."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    # work at a fixed size for stable thresholds
    scale = 800 / max(img.shape[:2])
    img = cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    bin_hull, bin_mask = detect_bin(img)
    if bin_hull is None:
        print('Bin not found')
        return None

    mk = red_mask(img, bin_mask)
    label, info, mhull = classify_marker(mk)

    out = img.copy()
    cv2.drawContours(out, [bin_hull], -1, (0, 255, 0), 2)
    if mhull is not None:
        cv2.drawContours(out, [mhull], -1, (255, 0, 0), 2)
    x, y, w, h = cv2.boundingRect(bin_hull)
    cv2.putText(out, f'Marker: {label}', (max(x, 10), max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    print('Marker:', label, info)
    if debug:
        cv2.imwrite('debug_bin_mask.png', bin_mask)
        cv2.imwrite('debug_red_mask.png', mk)
    cv2.imwrite('result.png', out)
    return label


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], debug='--debug' in sys.argv)

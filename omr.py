from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from document_scanner import scan_document


PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
MARKER_SIZE = 70
MARKER_MARGIN = 75
MARKER_CENTERS = np.array(
    [
        [MARKER_MARGIN + MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
        [MARKER_MARGIN + MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
    ],
    dtype=np.float32,
)

QUESTION_COUNT = 20
CHOICES = ("A", "B", "C", "D")
QUESTION_START_Y = 430
QUESTION_GAP_Y = 78
BUBBLE_RADIUS = 28
INNER_RADIUS = 17


def choices_for_count(count: int) -> tuple[str, ...]:
    """Return A..H style choice labels. The UI currently supports 2-8 choices."""
    count = int(count)
    if not 2 <= count <= 8:
        raise ValueError("Choice count must be between 2 and 8.")
    return tuple(chr(ord("A") + i) for i in range(count))


def bubble_x_for_choices(count: int) -> np.ndarray:
    """Keep the original A-D coordinates, while expanding cleanly up to 8 choices."""
    count = int(count)
    if not 2 <= count <= 8:
        raise ValueError("Choice count must be between 2 and 8.")
    center = 1005.0
    spacing = min(210.0, 840.0 / max(count - 1, 1))
    start = center - spacing * (count - 1) / 2.0
    return np.rint(start + np.arange(count) * spacing).astype(np.int32)


BUBBLE_X = bubble_x_for_choices(len(CHOICES))


@dataclass(frozen=True)
class QuestionDetection:
    question: int
    densities: tuple[float, ...]
    selected: str | None
    status: str
    confidence: float


@dataclass(frozen=True)
class ScanResult:
    aligned_bgr: np.ndarray
    threshold: np.ndarray
    detections: tuple[QuestionDetection, ...]
    marker_detection_used: bool
    warning: str | None = None
    scanner_bgr: np.ndarray | None = None
    scanner_page_found: bool = False
    choices: tuple[str, ...] = CHOICES
    bubble_rows: tuple[tuple[tuple[int, int], ...], ...] = ()
    grid_refinement_used: bool = False
    grid_confidence: float = 0.0
    residual_angle_deg: float = 0.0
    blur_score: float = 0.0


class OMRProcessingError(RuntimeError):
    pass


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise OMRProcessingError("The uploaded file could not be decoded as an image.")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _find_registration_markers(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    image_area = float(h * w)
    candidates: list[tuple[float, float, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.00035 or area > image_area * 0.05:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw == 0 or ch == 0:
            continue
        aspect = cw / float(ch)
        extent = area / float(cw * ch)
        if not 0.70 <= aspect <= 1.30 or extent < 0.72:
            continue
        candidates.append((x + cw / 2.0, y + ch / 2.0, area))

    if len(candidates) < 4:
        return None

    corner_targets = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    chosen: list[tuple[float, float]] = []
    used: set[int] = set()
    for target in corner_targets:
        scored: list[tuple[float, int]] = []
        for index, (cx, cy, area) in enumerate(candidates):
            if index in used:
                continue
            dist = float(np.linalg.norm(np.array([cx, cy]) - target))
            scored.append((dist - np.sqrt(area) * 0.20, index))
        if not scored:
            return None
        _, best_index = min(scored)
        used.add(best_index)
        cx, cy, _ = candidates[best_index]
        chosen.append((cx, cy))

    ordered = _order_points(np.array(chosen, dtype=np.float32))
    tl, tr, br, bl = ordered
    horizontal = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    vertical = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if horizontal < w * 0.45 or vertical < h * 0.45:
        return None
    return ordered


def _fallback_page_warp(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            continue
        source = _order_points(approx.reshape(4, 2))
        destination = np.array(
            [[0, 0], [PAGE_WIDTH - 1, 0], [PAGE_WIDTH - 1, PAGE_HEIGHT - 1], [0, PAGE_HEIGHT - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(image_bgr, matrix, (PAGE_WIDTH, PAGE_HEIGHT)), (
            "Registration squares were not found; the app used the page boundary instead. "
            "For the most reliable grading, keep all four black corner squares visible."
        )

    resized = cv2.resize(image_bgr, (PAGE_WIDTH, PAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized, (
        "The app could not detect the corner squares or page boundary, so it resized the image directly. "
        "Results may be inaccurate; retake the photo with the full sheet visible."
    )


def align_sheet(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str | None]:
    markers = _find_registration_markers(image_bgr)
    if markers is None:
        aligned, warning = _fallback_page_warp(image_bgr)
        return aligned, False, warning

    matrix = cv2.getPerspectiveTransform(markers, MARKER_CENTERS)
    aligned = cv2.warpPerspective(
        image_bgr,
        matrix,
        (PAGE_WIDTH, PAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, True, None


def _prepare_gray(aligned_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    # Remove remaining slow illumination gradients without erasing pencil/ink.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    flat = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return cv2.bilateralFilter(flat, 5, 22, 22)


def _prepare_threshold(aligned_bgr: np.ndarray) -> np.ndarray:
    gray = _prepare_gray(aligned_bgr)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        11,
    )
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return threshold


def bubble_centers(*, choice_count: int = 4, question_count: int = QUESTION_COUNT) -> list[list[tuple[int, int]]]:
    x_positions = bubble_x_for_choices(choice_count)
    rows: list[list[tuple[int, int]]] = []
    for question_index in range(question_count):
        y = QUESTION_START_Y + question_index * QUESTION_GAP_Y
        rows.append([(int(x), int(y)) for x in x_positions])
    return rows


def _circle_candidates(threshold: np.ndarray) -> np.ndarray:
    """Find printed/filled bubble-like contours. Filled bubbles are allowed."""
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    points: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 250 <= area <= 6200:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not 28 <= w <= 85 or not 28 <= h <= 85:
            continue
        aspect = w / float(max(h, 1))
        if not 0.68 <= aspect <= 1.47:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.42:
            continue
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        # Limit search to the answer-table region so letters/numbers elsewhere do not dominate.
        if 500 <= cx <= 1490 and 390 <= cy <= 1980:
            points.append((cx, cy))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _refine_grid(
    threshold: np.ndarray,
    expected_rows: list[list[tuple[int, int]]],
) -> tuple[list[list[tuple[int, int]]], bool, float, float]:
    """Use actual bubble circles as a second alignment system after corner-marker warping.

    RANSAC estimates a small translation/rotation/scale transform from expected bubble
    centers to observed bubble centers. This rejects handwriting and random blobs.
    """
    expected = np.asarray([p for row in expected_rows for p in row], dtype=np.float32)
    candidates = _circle_candidates(threshold)
    if len(candidates) < 6:
        return expected_rows, False, 0.0, 0.0

    matched_expected: list[np.ndarray] = []
    matched_actual: list[np.ndarray] = []
    used: set[int] = set()

    # Marker alignment should already be close. Only accept nearby candidates.
    for exp in expected:
        distances = np.linalg.norm(candidates - exp, axis=1)
        order = np.argsort(distances)
        picked = None
        for idx in order:
            idx = int(idx)
            if idx in used:
                continue
            if float(distances[idx]) <= 46.0:
                picked = idx
                break
        if picked is not None:
            used.add(picked)
            matched_expected.append(exp)
            matched_actual.append(candidates[picked])

    min_matches = max(6, int(round(len(expected) * 0.12)))
    if len(matched_expected) < min_matches:
        return expected_rows, False, len(matched_expected) / max(len(expected), 1), 0.0

    src = np.asarray(matched_expected, dtype=np.float32)
    dst = np.asarray(matched_actual, dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return expected_rows, False, 0.0, 0.0

    inlier_ratio = float(np.mean(inliers.ravel() > 0))
    coverage = len(matched_expected) / max(len(expected), 1)
    confidence = float(np.clip(0.65 * inlier_ratio + 0.35 * min(1.0, coverage / 0.45), 0.0, 1.0))
    if confidence < 0.52:
        return expected_rows, False, confidence, 0.0

    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.sqrt(a * a + b * b))
    # Reject transforms that are too large for a fine-alignment stage.
    if not 0.94 <= scale <= 1.06 or abs(angle) > 4.5:
        return expected_rows, False, confidence * 0.5, angle

    ones = np.ones((expected.shape[0], 1), dtype=np.float32)
    refined = np.hstack([expected, ones]) @ matrix.T
    refined = np.rint(refined).astype(np.int32)

    rows: list[list[tuple[int, int]]] = []
    k = 0
    for row in expected_rows:
        new_row = []
        for _ in row:
            x, y = refined[k]
            new_row.append((int(x), int(y)))
            k += 1
        rows.append(new_row)
    return rows, True, confidence, angle


def _density_for_circle(
    gray: np.ndarray,
    threshold: np.ndarray,
    center: tuple[int, int],
    radius: int = INNER_RADIUS,
) -> float:
    """Multi-signal fill score. Printed outlines are mostly outside the inner mask."""
    cx, cy = center
    h, w = gray.shape
    if not (radius + 3 <= cx < w - radius - 3 and radius + 3 <= cy < h - radius - 3):
        return 0.0

    y0, y1 = cy - radius, cy + radius + 1
    x0, x1 = cx - radius, cx + radius + 1
    local_gray = gray[y0:y1, x0:x1]
    local_thr = threshold[y0:y1, x0:x1]
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (xx * xx + yy * yy) <= radius * radius
    if not np.any(mask):
        return 0.0

    binary_density = float(np.mean(local_thr[mask] > 0))
    darkness = float(np.mean((255.0 - local_gray[mask]) / 255.0))

    # A fixed dark-pixel view helps with dense black/blue ink after normalization.
    fixed_density = float(np.mean(local_gray[mask] < 178))

    # Local paper reference: compare the bubble center to an annulus around it.
    outer = 38
    oy0, oy1 = max(0, cy - outer), min(h, cy + outer + 1)
    ox0, ox1 = max(0, cx - outer), min(w, cx + outer + 1)
    patch = gray[oy0:oy1, ox0:ox1]
    pyy, pxx = np.ogrid[oy0 - cy:oy1 - cy, ox0 - cx:ox1 - cx]
    rr2 = pxx * pxx + pyy * pyy
    annulus = (rr2 >= 31 * 31) & (rr2 <= 38 * 38)
    bg_mean = float(np.mean(patch[annulus])) if np.any(annulus) else 245.0
    inner_mean = float(np.mean(local_gray[mask]))
    contrast = float(np.clip((bg_mean - inner_mean) / 150.0, 0.0, 1.0))

    return float(np.clip(0.45 * binary_density + 0.20 * darkness + 0.20 * fixed_density + 0.15 * contrast, 0.0, 1.0))


def classify_densities(
    densities: Sequence[float],
    *,
    answer_key_mode: bool,
    choices: Sequence[str] = CHOICES,
) -> tuple[str | None, str, float]:
    """
    Dominance-first decision logic.

    The scanner no longer returns "unclear" simply because a mark is weaker than
    an old absolute threshold. Once a row is not blank, the strongest two
    choices are compared mathematically.

    Decision:
      * truly empty row -> blank
      * top two marks genuinely close -> multiple
      * otherwise -> choose the dominant (strongest) bubble

    This is intentionally based on BOTH absolute difference and relative
    difference so it works for light pencil as well as dark pen.
    """
    values = np.asarray(densities, dtype=np.float32)
    if len(values) < 2:
        return None, "unclear", 0.0

    order = np.argsort(values)[::-1]
    top_index = int(order[0])
    second_index = int(order[1])

    top = float(values[top_index])
    second = float(values[second_index])

    gap = max(0.0, top - second)
    ratio = top / max(second, 0.020)

    # Difference relative to the winner. 0 = exact tie, 1 = second is nearly zero.
    relative_gap = gap / max(top, 0.020)

    # Keep a true blank guard so normal paper / printed bubble rings do not
    # generate a forced answer.
    blank_threshold = 0.075 if answer_key_mode else 0.065
    if top < blank_threshold:
        return None, "blank", 0.0

    # A second mark must contain enough actual ink before it can create a
    # "multiple" result. This prevents two nearly-empty bubbles from looking
    # like a double answer because of print/lighting noise.
    second_has_ink = second >= (0.090 if answer_key_mode else 0.080)

    # "Very close" means that the runner-up is almost as intense as the winner.
    # We use two complementary tests:
    #   1) very small absolute difference
    #   2) runner-up is at least ~82% of winner
    # Requiring actual ink in the second bubble avoids false multiple marks.
    very_close = (
        gap <= (0.030 if answer_key_mode else 0.027)
        or relative_gap <= 0.18
        or ratio <= 1.20
    )

    if second_has_ink and very_close:
        # Tie confidence is intentionally low; UI will ask the teacher to review
        # it as a multiple mark.
        separation = np.clip(relative_gap / 0.18, 0.0, 1.0)
        confidence = float(separation * 45.0)
        return None, "multiple", confidence

    # Otherwise the strongest mark wins, including lighter marks that the old
    # code called "unclear". Confidence reflects how strongly it dominates.
    abs_score = np.clip(gap / 0.14, 0.0, 1.0)
    rel_score = np.clip((relative_gap - 0.18) / 0.52, 0.0, 1.0)
    ratio_score = np.clip((ratio - 1.20) / 1.20, 0.0, 1.0)
    confidence = float((0.45 * abs_score + 0.35 * rel_score + 0.20 * ratio_score) * 100.0)

    # Do not let a mathematically decisive but light mark display as 0% confidence.
    # The answer is still selected because its dominance is clear.
    confidence = max(confidence, 35.0)

    return str(choices[top_index]), "clear", confidence

def _robust_empty_baseline(all_density_rows: Sequence[Sequence[float]]) -> tuple[float, float]:
    """
    Estimate the sheet's typical empty-bubble level from the lower portion of
    observed bubble densities. Returns (median_empty, robust_sigma).
    """
    flat = []
    for row in all_density_rows:
        for v in row:
            try:
                flat.append(float(v))
            except Exception:
                pass
    if not flat:
        return 0.05, 0.02

    arr = np.asarray(flat, dtype=np.float32)
    # Use lower 55% as probable empty-bubble population.
    cutoff = np.percentile(arr, 55)
    empties = arr[arr <= cutoff]
    if empties.size < 6:
        empties = arr

    med = float(np.median(empties))
    mad = float(np.median(np.abs(empties - med)))
    sigma = max(0.008, 1.4826 * mad)
    return med, sigma


def _mark_texture_features(gray: np.ndarray, cx: int, cy: int, radius: int) -> dict:
    """
    Inspect the bubble interior for:
      - darkness
      - texture / patchiness
      - strong diagonal strokes that can indicate a cross-out
    """
    h, w = gray.shape[:2]
    r = max(4, int(radius * 0.72))
    x0, x1 = max(0, cx-r), min(w, cx+r+1)
    y0, y1 = max(0, cy-r), min(h, cy+r+1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return {"patchiness": 0.0, "diag_strength": 0.0, "center_dark": 0.0}

    # circular mask
    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    mx, my = (roi.shape[1]-1)/2.0, (roi.shape[0]-1)/2.0
    rr = min(roi.shape[:2]) * 0.45
    mask = (xx-mx)**2 + (yy-my)**2 <= rr**2
    vals = roi[mask].astype(np.float32)
    if vals.size == 0:
        return {"patchiness": 0.0, "diag_strength": 0.0, "center_dark": 0.0}

    center_dark = float(np.mean((255.0 - vals) / 255.0))
    patchiness = float(np.std(vals) / 128.0)
    patchiness = float(np.clip(patchiness, 0.0, 1.0))

    # Hough lines in bubble interior for crossed-out marks
    edges = cv2.Canny(roi, 60, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=max(6, int(radius*0.45)),
        minLineLength=max(6, int(radius*0.75)),
        maxLineGap=max(2, int(radius*0.18))
    )
    diag = 0.0
    if lines is not None:
        for ln in lines[:, 0, :]:
            x1l, y1l, x2l, y2l = map(float, ln)
            dx = x2l-x1l
            dy = y2l-y1l
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            # diagonal-ish line, not horizontal/vertical printed ring fragments
            if 20 <= angle <= 70 or 110 <= angle <= 160:
                diag = max(diag, min(1.0, length / max(1.0, radius*1.7)))

    return {
        "patchiness": patchiness,
        "diag_strength": float(diag),
        "center_dark": center_dark,
    }


def classify_densities_adaptive(
    densities: Sequence[float],
    *,
    empty_baseline: float,
    empty_sigma: float,
    answer_key_mode: bool,
    choices: Sequence[str] = CHOICES,
) -> tuple[str | None, str, float]:
    """
    Sheet-adaptive version of classify_densities.

    A true answer must rise above the current sheet's empty-bubble population.
    Once it does, dominance decides the winner; only close top-two marks are
    considered multiple.
    """
    values = np.asarray(densities, dtype=np.float32)
    if len(values) < 2:
        return None, "unclear", 0.0

    order = np.argsort(values)[::-1]
    top_index = int(order[0])
    second_index = int(order[1])

    top = float(values[top_index])
    second = float(values[second_index])
    gap = max(0.0, top - second)
    ratio = top / max(second, 0.020)
    relative_gap = gap / max(top, 0.020)

    # Dynamic blank cutoff based on actual sheet noise + conservative floor.
    floor = 0.070 if answer_key_mode else 0.060
    adaptive_blank = max(floor, empty_baseline + 2.8 * empty_sigma)
    if top < adaptive_blank:
        return None, "blank", 0.0

    second_has_ink = second >= max(
        0.080 if not answer_key_mode else 0.090,
        empty_baseline + 2.2 * empty_sigma,
    )

    very_close = (
        gap <= (0.030 if answer_key_mode else 0.027)
        or relative_gap <= 0.18
        or ratio <= 1.20
    )

    if second_has_ink and very_close:
        separation = np.clip(relative_gap / 0.18, 0.0, 1.0)
        return None, "multiple", float(separation * 45.0)

    abs_score = np.clip(gap / 0.14, 0.0, 1.0)
    rel_score = np.clip((relative_gap - 0.18) / 0.52, 0.0, 1.0)
    ratio_score = np.clip((ratio - 1.20) / 1.20, 0.0, 1.0)
    confidence = float((0.45 * abs_score + 0.35 * rel_score + 0.20 * ratio_score) * 100.0)
    confidence = max(confidence, 35.0)
    return str(choices[top_index]), "clear", confidence

def _blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def scan_sheet(
    image_bgr: np.ndarray,
    *,
    answer_key_mode: bool,
    scanner_mode: bool = False,
    choice_count: int = 4,
) -> ScanResult:
    choices = choices_for_count(choice_count)
    scanner_bgr = None
    scanner_page_found = False
    scanner_warning = None

    working_image = image_bgr
    if scanner_mode:
        scanned = scan_document(image_bgr, output_size=(PAGE_WIDTH, PAGE_HEIGHT))
        scanner_bgr = scanned.image_bgr
        scanner_page_found = scanned.page_found
        working_image = scanned.image_bgr
        if not scanned.page_found:
            scanner_warning = scanned.message

    aligned, marker_used, warning = align_sheet(working_image)
    if scanner_warning and warning:
        warning = scanner_warning + " " + warning
    elif scanner_warning:
        warning = scanner_warning

    threshold = _prepare_threshold(aligned)
    gray = _prepare_gray(aligned)
    expected_rows = bubble_centers(choice_count=choice_count)
    refined_rows, grid_used, grid_conf, residual_angle = _refine_grid(threshold, expected_rows)
    rows_to_use = refined_rows if grid_used else expected_rows

    blur = _blur_score(aligned)
    if blur < 38.0:
        blur_warning = "The image is quite blurry; answer confidence may be reduced. Retake the photo if possible."
        warning = (warning + " " + blur_warning) if warning else blur_warning

    if marker_used and not grid_used:
        grid_warning = "Bubble-grid refinement was not confident enough, so grading used the marker-aligned template positions."
        warning = (warning + " " + grid_warning) if warning else grid_warning

    detections: list[QuestionDetection] = []
    for q_index, row in enumerate(rows_to_use, start=1):
        densities = tuple(_density_for_circle(gray, threshold, center) for center in row)
        selected, status, confidence = classify_densities(
            densities,
            answer_key_mode=answer_key_mode,
            choices=choices,
        )
        detections.append(
            QuestionDetection(
                question=q_index,
                densities=densities,
                selected=selected,
                status=status,
                confidence=confidence,
            )
        )

    return ScanResult(
        aligned_bgr=aligned,
        threshold=threshold,
        detections=tuple(detections),
        marker_detection_used=marker_used,
        warning=warning,
        scanner_bgr=scanner_bgr,
        scanner_page_found=scanner_page_found,
        choices=choices,
        bubble_rows=tuple(tuple(row) for row in rows_to_use),
        grid_refinement_used=grid_used,
        grid_confidence=grid_conf,
        residual_angle_deg=residual_angle,
        blur_score=blur,
    )


def crop_question(aligned_bgr: np.ndarray, question_number: int) -> np.ndarray:
    index = question_number - 1
    y = QUESTION_START_Y + index * QUESTION_GAP_Y
    y1 = max(0, y - 38)
    y2 = min(PAGE_HEIGHT, y + 38)
    x1 = 430
    x2 = 1500
    return aligned_bgr[y1:y2, x1:x2].copy()


def draw_overlay(
    aligned_bgr: np.ndarray,
    detections: Sequence[QuestionDetection],
    *,
    key_answers: Sequence[str] | None = None,
    overrides: dict[int, str | None] | None = None,
    bubble_rows: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> np.ndarray:
    overlay = aligned_bgr.copy()
    overrides = overrides or {}

    for detection in detections:
        q = detection.question
        y = QUESTION_START_Y + (q - 1) * QUESTION_GAP_Y
        x1, x2 = 430, 1500
        y1, y2 = y - 34, y + 34

        selected = overrides.get(q, detection.selected)
        if selected == "Blank":
            selected = None

        if key_answers is None:
            if detection.status == "clear" or q in overrides:
                color = (50, 170, 70)
            elif detection.status == "blank":
                color = (145, 145, 145)
            else:
                color = (0, 165, 255)
        else:
            if detection.status not in {"clear"} and q not in overrides:
                color = (0, 165, 255)
            elif selected is None:
                color = (145, 145, 145)
            elif selected == key_answers[q - 1]:
                color = (50, 170, 70)
            else:
                color = (45, 45, 220)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label = selected if selected is not None else detection.status.upper()
        cv2.putText(
            overlay,
            f"Q{q}: {label}",
            (1505, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        if bubble_rows is not None and q - 1 < len(bubble_rows):
            for center in bubble_rows[q - 1]:
                cv2.circle(overlay, tuple(map(int, center)), BUBBLE_RADIUS + 4, (210, 120, 30), 1, cv2.LINE_AA)
                cv2.circle(overlay, tuple(map(int, center)), 3, (210, 120, 30), -1, cv2.LINE_AA)

    return overlay


def answers_from_detections(
    detections: Sequence[QuestionDetection],
    overrides: dict[int, str | None] | None = None,
) -> list[str | None]:
    overrides = overrides or {}
    answers: list[str | None] = []
    for detection in detections:
        value = overrides.get(detection.question, detection.selected)
        if value == "Blank":
            value = None
        answers.append(value)
    return answers


def grade_answers(student_answers: Sequence[str | None], key_answers: Sequence[str]) -> dict[str, object]:
    if len(student_answers) != len(key_answers):
        raise ValueError("Student answers and answer key must have the same number of questions.")

    rows: list[dict[str, object]] = []
    correct = 0
    blank = 0
    for index, (student, correct_answer) in enumerate(zip(student_answers, key_answers), start=1):
        is_correct = student == correct_answer
        if is_correct:
            correct += 1
        if student is None:
            blank += 1
        rows.append(
            {
                "Question": index,
                "Student answer": student or "Blank",
                "Correct answer": correct_answer,
                "Result": "Correct" if is_correct else ("Blank" if student is None else "Incorrect"),
            }
        )

    total = len(key_answers)
    return {
        "score": correct,
        "total": total,
        "percentage": (correct / total * 100.0) if total else 0.0,
        "blank": blank,
        "incorrect": total - correct - blank,
        "rows": rows,
    }


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise OMRProcessingError("Could not encode the processed image.")
    return encoded.tobytes()


def detection_rows(
    detections: Iterable[QuestionDetection],
    *,
    choices: Sequence[str] = CHOICES,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detection in detections:
        row: dict[str, object] = {
            "Question": detection.question,
            "Detected": detection.selected or detection.status.title(),
            "Status": detection.status.title(),
            "Confidence": round(detection.confidence, 1),
        }
        for choice, density in zip(choices, detection.densities):
            row[f"{choice} density"] = round(density, 3)
        rows.append(row)
    return rows

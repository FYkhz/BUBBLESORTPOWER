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
    """
    Strong paper/background normalization.

    Goal:
      - white paper stays white
      - gray paper is pushed toward white
      - slow shadows / camera exposure gradients are removed
      - pencil/ink remains dark

    This is intentionally illumination-invariant because OMR should care
    about ink relative to the nearby paper, not the absolute page brightness.
    """
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

    # Mild denoise first so the background estimate is not affected by sensor noise.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Estimate the slowly-changing paper background.
    # The kernel is much larger than a bubble, so ink/printed marks do not
    # become part of the background model.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=48, sigmaY=48)

    # Division normalization removes multiplicative lighting variation:
    # a gray/shadowed area becomes comparable to a white area.
    flat = cv2.divide(gray, np.maximum(bg, 1), scale=255)

    # Robust percentile stretch. Avoid min/max because one black mark can
    # otherwise distort the entire sheet.
    lo = float(np.percentile(flat, 1.5))
    hi = float(np.percentile(flat, 98.5))

    if hi - lo >= 10.0:
        flat = np.clip((flat.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
        flat = flat.astype(np.uint8)
    else:
        flat = flat.astype(np.uint8)

    # Gentle local contrast improves pencil without making gray paper "ink".
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(12, 12))
    flat = clahe.apply(flat)

    # Final edge-preserving smoothing.
    flat = cv2.bilateralFilter(flat, 5, 24, 24)

    return flat


def _prepare_threshold(aligned_bgr: np.ndarray) -> np.ndarray:
    """
    Threshold after background normalization.

    Two adaptive views are combined so light pencil survives while broad gray
    background regions do not become foreground.
    """
    gray = _prepare_gray(aligned_bgr)

    adaptive_gauss = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        45,
        10,
    )

    adaptive_mean = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        45,
        12,
    )

    # Require support from at least one adaptive method, but clean isolated noise.
    threshold = cv2.bitwise_or(adaptive_gauss, adaptive_mean)

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )

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
    """
    Illumination-invariant bubble fill score.

    Instead of asking:
        "How dark is this bubble absolutely?"

    we ask:
        "How much darker is this bubble than the paper immediately around it?"

    This makes gray paper, shadows, and exposure changes much less important.
    """
    cx, cy = center
    h, w = gray.shape

    outer_radius = max(40, radius + 18)

    if not (
        outer_radius + 2 <= cx < w - outer_radius - 2
        and outer_radius + 2 <= cy < h - outer_radius - 2
    ):
        return 0.0

    # --------------------------------------------------------------
    # Extract a local patch around the entire bubble.
    # --------------------------------------------------------------
    y0, y1 = cy - outer_radius, cy + outer_radius + 1
    x0, x1 = cx - outer_radius, cx + outer_radius + 1

    patch_gray = gray[y0:y1, x0:x1]
    patch_thr = threshold[y0:y1, x0:x1]

    yy, xx = np.ogrid[
        -outer_radius:outer_radius + 1,
        -outer_radius:outer_radius + 1,
    ]

    rr2 = xx * xx + yy * yy

    # Use a smaller inner region so the printed bubble outline contributes less.
    inner_radius = max(5, int(round(radius * 0.78)))
    inner_mask = rr2 <= inner_radius * inner_radius

    # Paper reference ring outside the printed bubble.
    annulus_inner = max(radius + 8, int(round(radius * 1.35)))
    annulus_outer = outer_radius - 2
    bg_mask = (
        (rr2 >= annulus_inner * annulus_inner)
        & (rr2 <= annulus_outer * annulus_outer)
    )

    if not np.any(inner_mask) or not np.any(bg_mask):
        return 0.0

    inner_gray = patch_gray[inner_mask].astype(np.float32)
    bg_gray = patch_gray[bg_mask].astype(np.float32)

    # --------------------------------------------------------------
    # Robust local paper statistics.
    # Median is safer than mean when handwriting or a line crosses nearby.
    # --------------------------------------------------------------
    bg_median = float(np.median(bg_gray))
    bg_mad = float(np.median(np.abs(bg_gray - bg_median)))
    bg_sigma = max(3.0, 1.4826 * bg_mad)

    inner_mean = float(np.mean(inner_gray))
    inner_median = float(np.median(inner_gray))

    # --------------------------------------------------------------
    # Signal 1: local contrast relative to nearby paper.
    # --------------------------------------------------------------
    contrast_mean = max(0.0, bg_median - inner_mean)
    contrast_median = max(0.0, bg_median - inner_median)

    contrast_score = float(
        np.clip(
            (0.65 * contrast_mean + 0.35 * contrast_median) / 95.0,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------------
    # Signal 2: locally dark pixels.
    #
    # Threshold is RELATIVE to the surrounding paper, not a fixed gray value.
    # On gray paper, this threshold shifts automatically.
    # --------------------------------------------------------------
    local_dark_cut = bg_median - max(12.0, 2.2 * bg_sigma)
    local_dark_density = float(np.mean(inner_gray <= local_dark_cut))

    # --------------------------------------------------------------
    # Signal 3: adaptive binary mask density.
    # --------------------------------------------------------------
    binary_density = float(np.mean(patch_thr[inner_mask] > 0))

    # --------------------------------------------------------------
    # Signal 4: center-core darkness.
    #
    # Useful for solid fills; keeps the printed ring from dominating.
    # --------------------------------------------------------------
    core_radius = max(4, int(round(radius * 0.48)))
    core_mask = rr2 <= core_radius * core_radius
    core_gray = patch_gray[core_mask].astype(np.float32)

    core_mean = float(np.mean(core_gray))
    core_contrast = max(0.0, bg_median - core_mean)
    core_score = float(np.clip(core_contrast / 90.0, 0.0, 1.0))

    # --------------------------------------------------------------
    # Signal 5: background-normalized darkness.
    #
    # This ratio is stable even when the page is globally gray.
    # --------------------------------------------------------------
    normalized_darkness = float(
        np.clip(
            (bg_median - inner_mean) / max(bg_median, 35.0),
            0.0,
            1.0,
        )
    )

    # Weighted ensemble.
    score = (
        0.28 * binary_density
        + 0.27 * local_dark_density
        + 0.22 * contrast_score
        + 0.15 * core_score
        + 0.08 * normalized_darkness
    )

    return float(np.clip(score, 0.0, 1.0))


def _robust_empty_baseline(
    all_density_rows: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """
    Learn the normal empty-bubble level from the CURRENT sheet.

    Most bubbles on a normal answer sheet are empty, so the lower
    part of the global score distribution is a strong empty reference.
    """
    flat: list[float] = []

    for row in all_density_rows:
        for value in row:
            try:
                flat.append(float(value))
            except Exception:
                pass

    if not flat:
        return 0.020, 0.010

    arr = np.asarray(flat, dtype=np.float32)

    cutoff = float(np.percentile(arr, 60))
    empties = arr[arr <= cutoff]

    if empties.size < max(6, int(arr.size * 0.25)):
        empties = arr

    median = float(np.median(empties))
    mad = float(np.median(np.abs(empties - median)))
    sigma = max(0.006, 1.4826 * mad)

    return median, sigma


def _smart_row_classification(
    densities: Sequence[float],
    *,
    answer_key_mode: bool,
    choices: Sequence[str],
    sheet_empty: float | None = None,
    sheet_sigma: float | None = None,
) -> tuple[str | None, str, float]:
    """
    Robust row + sheet OMR classifier.

    Handles:
      - blank rows
      - one light but obvious mark
      - one strong + faint erasure/smudge
      - two close selected bubbles
      - two unequal marks where one clearly wins
      - 3+ selected bubbles
      - all choices selected
      - raised empty-bubble scores caused by print/lighting
      - different pen/pencil darkness

    API stays unchanged:
        selected, status, confidence
    """
    values = np.asarray(densities, dtype=np.float32)

    if values.size < 2 or not np.all(np.isfinite(values)):
        return None, "unclear", 0.0

    n = len(values)
    order = np.argsort(values)[::-1]

    top_idx = int(order[0])
    second_idx = int(order[1])

    top = float(values[top_idx])
    second = float(values[second_idx])
    third = float(values[int(order[2])]) if n >= 3 else 0.0

    # ------------------------------------------------------------
    # Local empty reference.
    #
    # Use the lowest ~25% of this row, not half of the row.
    # This is important when 3 of 4 bubbles are intentionally marked.
    # ------------------------------------------------------------
    local_count = max(1, int(np.floor(n * 0.25)))
    local_low = np.sort(values)[:local_count]
    row_empty = float(np.median(local_low))

    if sheet_empty is None:
        sheet_empty = row_empty
    if sheet_sigma is None:
        sheet_sigma = 0.010

    sheet_empty = float(sheet_empty)
    noise = max(0.008, float(sheet_sigma))

    # Local contrast is best for deciding which bubbles in THIS row are
    # marked. Sheet contrast helps recognize "all choices selected".
    local_contrasts = values - row_empty
    sheet_contrasts = values - sheet_empty

    top_local = max(0.0, float(local_contrasts[top_idx]))
    second_local = max(0.0, float(local_contrasts[second_idx]))

    gap12 = max(0.0, top - second)
    gap23 = max(0.0, second - third)

    ratio12 = top / max(second, 0.010)
    relative_gap12 = gap12 / max(top, 0.010)

    row_min = float(np.min(values))
    row_max = float(np.max(values))
    row_spread = row_max - row_min

    # ------------------------------------------------------------
    # Special case: ALL / almost all bubbles are strongly elevated.
    #
    # When there is no low bubble inside the row, local contrast alone
    # cannot tell that everything was selected. Compare the entire row
    # to the sheet-wide empty baseline.
    # ------------------------------------------------------------
    uniformly_elevated = (
        row_min >= max(0.055, sheet_empty + max(0.040, 3.0 * noise))
        and row_spread <= 0.055
    )

    if uniformly_elevated:
        return None, "multiple", 90.0

    # ------------------------------------------------------------
    # Candidate and strong mark thresholds.
    # ------------------------------------------------------------
    candidate_local_floor = max(0.022, 1.8 * noise)
    strong_local_floor = max(0.036, 2.8 * noise)

    candidate_mask = (
        (values >= 0.040)
        & (local_contrasts >= candidate_local_floor)
    )

    strong_mask = (
        (values >= 0.055)
        & (local_contrasts >= strong_local_floor)
    )

    candidate_indices = np.where(candidate_mask)[0].tolist()
    strong_indices = np.where(strong_mask)[0].tolist()

    candidate_count = len(candidate_indices)
    strong_count = len(strong_indices)

    # ------------------------------------------------------------
    # BLANK
    #
    # Weak + flat + no candidate = blank.
    # A Q7-like [0,0,0.063,0] survives because its local contrast is large.
    # ------------------------------------------------------------
    top_small = top < max(
        0.055 if not answer_key_mode else 0.060,
        sheet_empty + 2.0 * noise,
    )

    no_relative_winner = (
        top_local < max(0.022, 1.8 * noise)
        and gap12 < 0.018
    )

    if top_small and no_relative_winner and candidate_count == 0:
        return None, "blank", 0.0

    # ------------------------------------------------------------
    # 3+ strong selections => Multiple.
    # ------------------------------------------------------------
    if strong_count >= 3:
        confidence = float(
            np.clip(
                70.0 + 120.0 * max(0.0, float(np.mean(values[strong_mask])) - row_empty),
                70.0,
                99.0,
            )
        )
        return None, "multiple", confidence

    # ------------------------------------------------------------
    # Exactly 2 strong selections.
    #
    # If close -> Multiple.
    # If clearly unequal -> strongest wins.
    # ------------------------------------------------------------
    if strong_count == 2:
        sidx = sorted(
            strong_indices,
            key=lambda i: float(values[i]),
            reverse=True,
        )

        s1 = float(values[sidx[0]])
        s2 = float(values[sidx[1]])

        pair_gap = s1 - s2
        pair_ratio = s1 / max(s2, 0.010)
        pair_relative_gap = pair_gap / max(s1, 0.010)

        close_pair = (
            pair_gap <= 0.040
            or pair_relative_gap <= 0.24
            or pair_ratio <= 1.32
        )

        other_indices = [
            i for i in range(n)
            if i not in strong_indices
        ]

        other_max = (
            float(np.max(values[other_indices]))
            if other_indices
            else row_empty
        )

        pair_separation = min(s1, s2) - other_max

        if (
            close_pair
            and pair_separation >= max(0.018, 1.6 * noise)
        ):
            closeness = 1.0 - np.clip(
                pair_relative_gap / 0.24,
                0.0,
                1.0,
            )

            separation = np.clip(
                pair_separation / 0.12,
                0.0,
                1.0,
            )

            confidence = float(
                np.clip(
                    (
                        0.60 * closeness
                        + 0.40 * separation
                    )
                    * 100.0,
                    60.0,
                    99.0,
                )
            )

            return None, "multiple", confidence

        # Two visible marks, but one clearly dominates.
        confidence = float(
            np.clip(
                55.0
                + 150.0 * pair_gap
                + 25.0 * min(1.0, top_local / 0.12),
                55.0,
                99.0,
            )
        )

        return str(choices[top_idx]), "clear", confidence

    # ------------------------------------------------------------
    # One strong selection.
    #
    # Normally pick it. Only call Multiple if a second lighter candidate
    # is genuinely close and clearly above the remaining choices.
    # ------------------------------------------------------------
    if strong_count == 1:
        if candidate_count >= 2:
            second_candidate_real = (
                second >= 0.045
                and second_local >= max(0.026, 2.0 * noise)
            )

            close_second = (
                gap12 <= 0.025
                or relative_gap12 <= 0.18
                or ratio12 <= 1.22
            )

            second_separated_from_third = (
                n < 3
                or gap23 >= 0.018
                or second >= third * 1.25
            )

            if (
                second_candidate_real
                and close_second
                and second_separated_from_third
            ):
                return None, "multiple", 60.0

        confidence = float(
            np.clip(
                55.0
                + 220.0 * top_local
                + 90.0 * gap12,
                55.0,
                99.0,
            )
        )

        return str(choices[top_idx]), "clear", confidence

    # ------------------------------------------------------------
    # No strong marks, but light-pencil candidates exist.
    # ------------------------------------------------------------
    if candidate_count >= 3:
        candidate_values = values[candidate_mask]

        if float(np.mean(candidate_values) - row_empty) >= max(
            0.026,
            2.0 * noise,
        ):
            return None, "multiple", 60.0

    if candidate_count == 2:
        cidx = sorted(
            candidate_indices,
            key=lambda i: float(values[i]),
            reverse=True,
        )

        c1 = float(values[cidx[0]])
        c2 = float(values[cidx[1]])

        cgap = c1 - c2
        cratio = c1 / max(c2, 0.010)
        crel = cgap / max(c1, 0.010)

        close_candidates = (
            cgap <= 0.025
            or crel <= 0.20
            or cratio <= 1.25
        )

        other_indices = [
            i for i in range(n)
            if i not in candidate_indices
        ]

        other_max = (
            float(np.max(values[other_indices]))
            if other_indices
            else row_empty
        )

        pair_sep = min(c1, c2) - other_max

        if (
            close_candidates
            and pair_sep >= max(0.016, 1.5 * noise)
        ):
            return None, "multiple", 58.0

        return str(choices[top_idx]), "clear", 50.0

    # ------------------------------------------------------------
    # One light but obvious relative winner.
    # ------------------------------------------------------------
    winner_dominates = (
        top_local >= max(0.022, 1.8 * noise)
        and (
            gap12 >= 0.020
            or ratio12 >= 1.35
            or relative_gap12 >= 0.26
        )
    )

    if winner_dominates:
        gap_score = np.clip(gap12 / 0.12, 0.0, 1.0)
        ratio_score = np.clip((ratio12 - 1.0) / 1.7, 0.0, 1.0)
        contrast_score = np.clip(top_local / 0.11, 0.0, 1.0)

        confidence = float(
            (
                0.35 * gap_score
                + 0.25 * ratio_score
                + 0.40 * contrast_score
            )
            * 100.0
        )

        return str(choices[top_idx]), "clear", max(
            confidence,
            55.0,
        )

    # ------------------------------------------------------------
    # Strange / uniformly ambiguous row.
    #
    # If at least two bubbles still stand above the local empty reference
    # but there is no clear winner, Multiple is safer than forcing one.
    # ------------------------------------------------------------
    elevated_mask = local_contrasts >= max(0.022, 1.8 * noise)
    elevated_count = int(np.sum(elevated_mask))

    if elevated_count >= 2:
        return None, "multiple", 55.0

    return None, "blank", 0.0


def classify_densities(
    densities: Sequence[float],
    *,
    answer_key_mode: bool,
    choices: Sequence[str] = CHOICES,
) -> tuple[str | None, str, float]:
    """
    Backwards-compatible row-only classifier.
    """
    return _smart_row_classification(
        densities,
        answer_key_mode=answer_key_mode,
        choices=choices,
    )


def classify_densities_adaptive(
    densities: Sequence[float],
    *,
    empty_baseline: float,
    empty_sigma: float,
    answer_key_mode: bool,
    choices: Sequence[str] = CHOICES,
) -> tuple[str | None, str, float]:
    """
    Preferred classifier used by scan_sheet().
    """
    return _smart_row_classification(
        densities,
        answer_key_mode=answer_key_mode,
        choices=choices,
        sheet_empty=empty_baseline,
        sheet_sigma=empty_sigma,
    )

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

    # Background diagnostics after normalization. A gray original is okay;
    # what matters is whether normalization produced a stable paper field.
    paper_level = float(np.percentile(gray, 75))
    paper_low = float(np.percentile(gray, 20))
    paper_spread = paper_level - paper_low

    if paper_level < 175:
        bg_warning = (
            "The page background is unusually dark. The scanner normalized it, "
            "but brighter/even lighting may improve reliability."
        )
        warning = (warning + " " + bg_warning) if warning else bg_warning

    if blur < 38.0:
        blur_warning = "The image is quite blurry; answer confidence may be reduced. Retake the photo if possible."
        warning = (warning + " " + blur_warning) if warning else blur_warning

    if marker_used and not grid_used:
        grid_warning = "Bubble-grid refinement was not confident enough, so grading used the marker-aligned template positions."
        warning = (warning + " " + grid_warning) if warning else grid_warning

    # Two-pass answer analysis:
    # 1) measure all bubbles
    # 2) learn this sheet's empty-bubble baseline
    # 3) classify each row using local + sheet-wide context
    measured_rows: list[tuple[float, ...]] = []

    for row in rows_to_use:
        densities = tuple(
            _density_for_circle(gray, threshold, center)
            for center in row
        )
        measured_rows.append(densities)

    sheet_empty, sheet_sigma = _robust_empty_baseline(measured_rows)

    detections: list[QuestionDetection] = []

    for q_index, densities in enumerate(measured_rows, start=1):
        selected, status, confidence = classify_densities_adaptive(
            densities,
            empty_baseline=sheet_empty,
            empty_sigma=sheet_sigma,
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

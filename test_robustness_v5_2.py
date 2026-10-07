import numpy as np
from omr import _robust_empty_baseline, classify_densities_adaptive
from scan_quality import blur_score, glare_fraction

rows = [
    [0.045, 0.050, 0.048, 0.052],
    [0.047, 0.049, 0.051, 0.046],
    [0.190, 0.050, 0.049, 0.048],
]
base, sigma = _robust_empty_baseline(rows)
assert 0.03 <= base <= 0.07
assert sigma >= 0.008

a, s, c = classify_densities_adaptive(
    [0.19, 0.052, 0.049, 0.048],
    empty_baseline=base,
    empty_sigma=sigma,
    answer_key_mode=False,
    choices=("A","B","C","D"),
)
assert a == "A" and s == "clear"

a, s, c = classify_densities_adaptive(
    [0.185, 0.176, 0.048, 0.050],
    empty_baseline=base,
    empty_sigma=sigma,
    answer_key_mode=False,
    choices=("A","B","C","D"),
)
assert a is None and s == "multiple"

img = np.full((200,300,3), 255, np.uint8)
assert glare_fraction(img) > 0.9
assert blur_score(img) == 0.0

print("Robustness v5.2 tests passed.")

# Robustness v5.2

This build adds safeguards for common real-world OMR failures that are easy to overlook.

## Added protections
- Per-sheet empty-bubble baseline utility
- Adaptive blank threshold helper
- Cross-out / erasure inspection helpers
- Blur scoring
- Glare detection
- Corner clipping risk check
- Existing SmartGrid geometry, perspective correction, multi-choice support and dominance logic are preserved

## Intended decision philosophy
- Do not force an answer on a truly blank row.
- Once a row contains real ink, compare the two strongest marks mathematically.
- If one clearly dominates, choose it.
- If the top two are genuinely close and both contain real ink, mark Multiple.
- Suspicious corrections/cross-outs can be flagged for review rather than silently misgraded.

These utilities are deliberately modular so they can be tuned without destabilizing the already-working SmartGrid scanner.

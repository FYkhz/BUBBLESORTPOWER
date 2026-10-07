# Smart OMR Scanner Upgrade

This build keeps the aggressive document scanner and adds a second geometry stage using the bubble grid itself.

## New in this build

- Four-corner registration + perspective correction
- Aggressive shadow/background cleanup
- Bubble-grid auto-alignment after the page is flattened
- RANSAC rejection of bad circle detections and handwriting noise
- Residual tilt/scale/translation correction without repeatedly warping the image
- Supports 2 to 8 choices per question (A-H)
- Multi-signal bubble fill scoring:
  - adaptive threshold
  - darkness
  - fixed dark-pixel view
  - local paper/background comparison
- Dominance scoring so the winning bubble must beat the second-best mark
- Blur warning
- Grid confidence and residual-angle display
- Pure-white generated answer sheet with Name section only
- Bubble-center diagnostic overlay

## Important

Use the same choice-count setting when:
1. generating the sheet,
2. creating the answer key,
3. grading students.

Changing the choice count clears the saved answer key so two incompatible layouts cannot be mixed.

## Tests included

- Original MVP synthetic perspective test
- New smart-grid test with camera-like perspective/rotation
- 4-choice layout
- 6-choice layout

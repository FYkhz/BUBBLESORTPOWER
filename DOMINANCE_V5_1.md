# Dominance v5.1

This update changes borderline answer handling.

## New decision rule

1. If no bubble contains enough ink, mark **Blank**.
2. Rank all bubble intensity scores.
3. Compare the strongest and second-strongest bubbles.
4. If the second bubble has real ink and the two scores are very close, mark **Multiple**.
5. Otherwise select the strongest bubble — even when the mark is light.

This removes most unnecessary "Unclear" outcomes while still protecting against true double marks.

The comparison uses:
- absolute intensity gap
- relative intensity gap
- strongest/second-strongest ratio

The existing SmartGrid alignment, aggressive scanner, multi-choice support, local-background scoring and perspective correction remain unchanged.

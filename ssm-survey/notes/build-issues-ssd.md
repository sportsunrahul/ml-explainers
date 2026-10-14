# Build issues found while building ch-ssd (2026-10-11)

- Mamba-2 Sec 6.3 (efficient.tex, "Center Blocks"): "Multiplying by the X values, which has cost BMM(T/Q, Q, P, N)". A (Q x Q) block times X (Q x P) is BMM(T/Q, Q, P, Q); the last index should be Q (equal to N only under Q = N). The page counts T Q P for that step (our reading).
- mamba2-mqar-n256-1024-d32 (0.52) is marked uncertain, so check-page refuses it in a span. ch-ssd now says "roughly half the queries" in words, labelled as read from the plot. ch-recall may hit the same failure.
- Fig 6 vs Sec 8.1 conv placement (already in notes): ch-ssd states the disagreement and draws per the Sec 8.1 equations.

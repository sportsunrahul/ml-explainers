# Build issues found while building ch-lti, ch-scan, ch-interlude, atlas scan

- claims/mamba.json `mamba-ind-mamba-1m` has the value "100", but Table 11 shows a check mark ("perfect"), not a number. The page avoids it.
- Mamba App. D ("Memory") gives the backward inputs as O(BLN + DN), but Sec 3.3.2 and App. D ("Speed") say O(BLD + DN). Probably a typo in the paper. The page uses BLD.
- page-chrome builds the rail's extras list only from `section.chapter` elements. The interlude is `section.interlude` (as instructed), so it is in the toc's "more" links but not in the rail extras. `class="chapter interlude"` would add it to the rail without changing the numbering, because numbering comes from `nav.toc ol > li` only.
- Mamba's text says the fused layer has "the same memory requirements" as FlashAttention, but Table 15 shows Mamba slightly higher at every batch size. The page says "comparable, not lower".

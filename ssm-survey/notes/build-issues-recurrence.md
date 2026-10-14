# Build issues from ch-recurrence (2026-10-11)

- notes/papers/lru.md Sec 8B and outline-feedback (lru + griffin) say LRU's sCIFAR is not comparable with S4's 91.13. A fact check found S4's 91.13 is most likely the same colored sequential CIFAR task (LRU App D.4; S4D App B, appendix_experiments.tex line 195, calls the grayscale task a variant of "the normal sequential CIFAR-10 task"). The page now labels the like-for-like reading as ours; lru.md and the claim text of lru-s4-paper-scifar should be updated to match.
- Atlas dialogs defined here, first in order so they win dedup: dlg-rnn, dlg-eigenvalue, dlg-complex, dlg-impulse-response, dlg-vanishing-gradient, dlg-parallel-scan, dlg-lra. Later chapters can reuse them via data-open.
- assets/atlas/build.mjs rejects non-ASCII in scenes.js, so shared core-*.js files must escape Greek letters (λ). Worth a line in build-brief.md.

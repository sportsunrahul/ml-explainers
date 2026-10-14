# MetaCLIP: Demystifying CLIP Data

## Identity
- arXiv 2309.16671, version read: v6 (v1 Thu 28 Sep 2023; v6 Sun 23 Nov 2025). Hu Xu ... Christoph Feichtenhofer (10 authors, FAIR Meta, NYU, UW).
- Venue: PDF header says "Preprint" (LaTeX uses the ICLR 2024 template; not stated as accepted in the source).
- Repo: https://github.com/facebookresearch/MetaCLIP

## Key idea
CLIP's edge is its data, not its loss. MetaCLIP rebuilds CLIP's 500,000-entry metadata, sub-string matches raw
CommonCrawl text to it, and caps each entry at t = 20k pairs, which flattens the head and keeps the tail; model and
training are frozen to the CLIP recipe so only data varies.

## Defining step (no loss equation; curation)
Entries with fewer than t pairs keep all; entries with more are sub-sampled to t = 20k (Sec. 3.4). Pool 1.6B pairs,
5.6B matches; 114k of 500k entries have no match (Sec. 3.2).

## Key number
ViT-B/16, 400M pairs, CLIP training setup: zero-shot ImageNet 70.8 vs CLIP (WIT400M, "our eval.") 68.3.
Table 4, ViT-B/16 rows, ImageNet column; also in the abstract. (CLIP paper's own Table 11 gives B/16 68.6; 68.3 is MetaCLIP's re-evaluation.)
Scaling to 1B data at same budget: 72.4 (abstract; ViT-B/16 per SigLIP 2 Table 1 row MetaCLIP B/16 72.4).

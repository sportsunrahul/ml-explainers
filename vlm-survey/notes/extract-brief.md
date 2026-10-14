# Extraction brief (shared by all extraction agents)

Project: ~/paper-explainers/vlm-survey. Read outline.md there first; your papers' chapters list
under "Need:" exactly what to extract. Extract ONLY what the outline needs (plus anything that
contradicts the outline). Do not summarise whole papers.

Sources: sources/<key>/src/*.tex (LaTeX; use this for equations and numbers), sources/<key>/paper.pdf
(tables/figures the LaTeX lacks; for PDFs > 20 MB read only specific pages with the Read tool's
`pages` param, or use `pdftotext -f N -l M`). SigLIP has no LaTeX: use pdftotext on its PDF.

Write two files per paper:
1. notes/papers/<key>.md -- for each chapter item: the mechanism in plain words, each needed
   equation copied in TeX with its equation number (or "unnumbered, Sec. X / Fig. Y pseudocode")
   and a variable-by-variable gloss, the exact numbers, and design details a visual depends on.
   End with a section "## Outline assertions I could not confirm or that are wrong" listing each
   with what the source actually says.
2. notes/claims/<key>.json -- a JSON array; every number you report is one entry:
   {"id": "<key>-<short-slug>", "claim": "...", "value": "...", "paper": "<name>", "version": "vN",
    "loc": "Table 3, row X / Sec. 3.2 para 2 / Eq. 4", "uncertain": false}
   Format example: ~/paper-explainers/diffusion-survey/notes/claims/beat-gans.json.
   Anything not verifiable from the primary source: "uncertain": true, with a note in "claim".
   Values must be verbatim from the paper (units included). Aim for claims the page will show
   (roughly 10-25 per paper), not every number.

Keys/versions: vit 2010.11929v2, clip 2103.00020v1, align 2102.05918v2, siglip 2303.15343v4,
blip 2201.12086v2, flamingo 2204.14198v2, blip2 2301.12597v3, llava 2304.08485v2,
llava15 2310.03744v2, qwenvl 2308.12966v3, florence2 2311.06242v1, qwen2vl 2409.12191v2,
idefics2 2405.02246v1, molmo 2409.17146v2, chameleon 2405.09818v2, qwen3vl 2511.21631v2.

Use only ASCII punctuation in files you write (no smart quotes, no em/en dashes).
Validate your JSON with `python3 -m json.tool`. Return a 5-10 line summary: files written, claim
count, and the unconfirmed/wrong outline assertions.

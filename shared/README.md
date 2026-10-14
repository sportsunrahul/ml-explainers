# Design pass on the six "explained in 3D" pages (2026-10-01)

Each page in `~/paper-explainers/<slug>/` was rebuilt from `<slug>/page.py`. Each page keeps its own fonts,
colors and 3D template. Per-page notes, including the equation transcriptions, are in `<slug>/notes/design.md`.

| slug | page |
|---|---|
| transformers | Transformers and LLMs, explained in 3D.html (the pilot) |
| dino | dino-explained.html |
| clip | How CLIP works, explained in 3D.html |
| post-training | Post-training and alignment, explained in 3D.html |
| seeing-tokens | Seeing in tokens_ visual tokenizers and VLMs.html |
| world-models | World models and video generation, explained in 3D.html |

## Rebuild one page
```bash
cd ~/paper-explainers/<slug> && python3 page.py
cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/prerender-math.mjs \
  ~/paper-explainers/<slug>/page.src.html "$HOME/paper-explainers/<slug>/<page file name>"
```

## Check one page
```bash
cd ~/paper-explainers/.node && node ~/paper-explainers/shared/qa.mjs "$HOME/paper-explainers/<slug>/<page>" ~/paper-explainers/<slug>/shots
node ~/paper-explainers/shared/smoke.mjs "$HOME/paper-explainers/<slug>/<page>"
```

## Revert one page
```bash
cp ~/paper-explainers/<slug>/original.html "$HOME/paper-explainers/<slug>/<page file name>"
```

## Shared pieces (`shared/`)
- `common.py`: section bar and progress line, KaTeX stylesheet link (cdnjs 0.18.9 with SRI), the `s.g(word)` glow
  hook in `stage()`, one left edge for all prose, and one list item per footer source.
- `chrome.css`, `chrome.js`: the section bar with a Sections menu when the links do not fit, and equation terms
  that light their figure.
- `qa.mjs`: screenshots and checks. `smoke.mjs`: clicks every control. `BRIEF.md`: the rules each page followed.

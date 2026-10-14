# Chapter-building brief (shared by all chapter agents)

Project: ~/paper-explainers/vlm-survey. You build one or two chapters of a single-page, intuition-first,
3Blue1Brown-style interactive explainer of vision-language models. The reader knows transformers and CNNs
(no attention or convolution basics) but wants everything else explained from scratch. Page only, no audio.

## Read first, in this order
1. outline.md (your chapter specs, THEN the two sections "Corrections from extraction" and "Added at
   checkpoint 2", which override the specs above them; the final chapter order and ids are listed there).
2. ~/.claude/skills/paper-explainer/references/page-authoring.md (the standard; follow it exactly).
3. parts/ch-shared-space.html + parts/viz-contrastive.js: the approved exemplar chapter. Match its tone,
   depth, structure (2-sentence setup -> figure -> h3 sections -> prose -> equation build -> gloss ->
   bridge to the next chapter), its prediction prompts, and its honesty about what is toy.
4. explainer.head.part: the global CSS and the `PE` toolkit (canvas, animate, instant, rng, label, mix,
   view, dot, arrow, axes, polyline, linkPulses, contrast, softmax). Use it; do not modify it.
5. notes/papers/<key>.md and notes/claims/<key>.json for your papers (facts come ONLY from these and the
   sources/ they cite; if you need a fact not there, read sources/<key>/src/*.tex or paper.pdf yourself).

## Page-wide color meanings (never reuse a color for anything else)
blue = vision (pixels, patches, image embeddings, encoder); yellow = language (text tokens, captions, LLM);
green = the bridge/connector, a correct match; red = loss, error, cost; brown = frozen weights, negatives,
background, padding.

## Files you may write (nothing else)
- parts/ch-<slug>.html (the section; start with `<p class="kicker">Act ...</p>` ONLY if your chapter opens
  an act: acts open at ch-shared-space (I), ch-bootstrap (II), ch-llava (III), ch-resolution (IV),
  ch-recipes (V), ch-early-fusion (VI))
- parts/viz-<slug>.js (a `<script>(() => { ... })();</script>` block, like viz-contrastive.js; one file per
  figure if you have two figures)
- notes/claims/page-<slug>.json for page-level claims (ids prefixed `page-<slug>-`), e.g. a number you
  derive from sourced values; same schema as the other claims files.
- build/fork-<slug>/ (your private test build) and build/sim/<slug>-*.cjs (simulations).
Never edit explainer.head.part, parts/body-top.html, parts/ORDER, build.sh, other chapters' parts, or any
existing notes/claims/<paper>.json (several agents cite the same paper files). If the claim lint fails on
wording, change your prose, not the claim.

## Rules
- Every id (figure, canvas, button, slider, readout, eq) is prefixed with your figure slug:
  viz-<slug>, <slug>-canvas, btn-<slug>-..., rng-<slug>-..., <slug>-readout, eq-<slug>...
- Helpers stay inside your IIFE. No `Object.assign(PE, ...)`, no global CSS; if you need CSS, scope it
  under `#viz-<slug>` in a `<style>` inside your section.
- Figures: `<figure class="viz" id="viz-<slug>" data-toy|data-computed|data-sourced>` + matching badge;
  canvas/SVG sized with PE.canvas (DPR-aware, ResizeObserver) and readable at 375 px; an aria-live
  readout; keyboard reachable; `PE.linkPulses(fig, redraw)` wired and actually used in drawing for every
  link word your equations use; before/after as two buttons, never a toggle; a `data-cue-reset` button;
  every state change tweened with PE.animate (which already respects instant()).
- Equations: TeX in `<div class="eq" id="eq-<slug>" data-build data-viz="viz-<slug>">` with
  `\htmlData{step=N, link=word}{...}` per term and `\htmlClass{t-blue}{...}` colours; equation number and
  paper version under it in `<p class="eqno">`; a `<ul class="gloss">` variable by variable. Intuition
  first, equation second.
- Numbers: every sourced number in prose/captions is inside `<span data-claim="<id>">`, and the span text
  must CONTAIN the claim's value verbatim (lint compares them; prefer number-only spans). Derived numbers
  go in `<span data-computed>`. Never show an `"uncertain": true` claim as fact.
- At least one "predict before you press" prompt per chapter; the visual must then answer it.
- Simulate any computed toy in node BEFORE writing prose (build/sim/pe.cjs exposes the page's PE:
  `require('./pe.cjs')` sets global.PE). Write the prose to what the toy actually does.
- ASCII punctuation only in code and prose: no smart quotes, no em or en dashes (and no &mdash; / &ndash;
  entities); use commas, colons or parentheses. Entities like &tau; &times; &asymp; are fine.
- Not everything needs to be a canvas; SVG or HTML/CSS grids are fine when clearer (e.g. token rows).

## Test (must PASS before you return)
```
cd ~/paper-explainers/vlm-survey
printf 'ch-<slug>.html\nviz-<slug>.js\n' > build/fork-<slug>.order     # list only your parts
bash build.sh build/fork-<slug>.order build/fork-<slug>
cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/vizshots.mjs \
  ~/paper-explainers/vlm-survey/build/fork-<slug>/explainer.html ~/paper-explainers/vlm-survey/build/fork-<slug>/viz \
  viz-<slug> btn-<slug>-a,btn-<slug>-b,... 1500
```
(mkdir build/fork-<slug>/viz first.) Fix until check-page prints PASS with no FAIL lines. READ the PNGs
(Read tool) in dark, light and mobile: no clipped or overlapping labels, nothing off-canvas, states look
right. Review the "bare numbers" list for any sourced number missing a data-claim.

## Return (10-15 lines)
Files written; QA result; the figure's states/buttons; claims added; anything in the outline you cut or
changed because the sources did not support it; one line on what the reader should now understand.

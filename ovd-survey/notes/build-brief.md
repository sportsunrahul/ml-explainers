# Chapter builder brief (read fully before writing code)

Project: `~/paper-explainers/ovd-survey/`. You build one group of chapters of an interactive, intuition-first
explainer page about open-vocabulary object detection, plus the matching atlas chapter(s). Chapter 1 and the
hero are done and are your reference for quality, style and code idiom. Read these first, in this order:

1. `outline.md`: the plan. Your chapters' specs, then EVERY correction section below them ("Corrections from
   extraction", the per-group paragraphs, and "Fact-check corrections", which wins over everything).
2. `notes/design.md`: colors (one meaning each), the tap strip, the detection overlay vocabulary, principles.
3. `~/.claude/skills/paper-explainer/references/page-authoring.md`: the 3Blue1Brown standard, equation builds,
   architecture walkthrough (archflow), claim cards, cue contract, design rules, 3D stages, atlas mode, honesty.
4. Reference code: `parts/ch-text-classifier.html` (chapter markup), `parts/scene-classifier.js` (a scene module
   shared by both pages), `parts/viz-classifier.js` (long-page wiring), `parts/lib-detkit.js` (the DK kit:
   C(), mixh, rr, brackets, tag, OBJ toy objects, WAREHOUSE scene, drawScene, boxOf, strip/tap strip),
   `explainer.head.part` (PE helpers and CSS classes), `atlas/chapters/reads.html` + `atlas/scenes/reads.js`.
   For archflow: the spec format is in the header of `parts/lib-archflow.js`; a worked example is the JEPA
   page: `~/paper-explainers/jepa-survey/parts/viz-ijflow.js` and its markup in `ch-ijepa.html` (search
   `viz-ijflow`). For 3D: the API header of `parts/lib-pe3.js`; examples `~/paper-explainers/jepa-survey/parts/viz-collapse.js`.
5. Your papers' notes in `notes/papers/<paper>.md` and claims in `notes/claims/<group>.json`; chart data in
   `notes/data/`.

## What you deliver (only these files; never edit anyone else's)
- `parts/ch-<id>.html` per chapter: `<section id="ch-<id>" class="chapter">` whose first child is the `<h2>`.
  Precede it with `<p class="kicker"><span class="act-n">Act N</span>Title</p>` when your chapter opens an act
  (the TOC in `parts/body-top.html` lists the acts and chapter ids; keep those ids exactly).
- Model chapters open (after the h2) with the tap strip: `<div class="tapstrip" data-model="vild|owl|glip|gdino|yolow" role="img" aria-label="..."></div>`
  then a one-line `<p class="note">` saying where text enters this model.
- `parts/scene-<name>.js` for any figure logic the atlas also uses (a factory like `SceneClassifier(el, o)`
  that takes `{ aspect, bottom, glow, onSay }` and returns `{ set..., reset, redraw, destroy }`), and
  `parts/viz-<name>.js` for long-page wiring (buttons, readout, `PE.linkPulses`). Pure long-page figures can
  live in one `viz-*.js` file. Every file is a `<script>...</script>` (or `<style>` + `<script>`) block.
  Name globals uniquely (prefix with your chapter id); wrap long-page wiring in an IIFE.
- `parts/ORDER.<group>`: your private build order: the library lines below, then your parts in page order.
  ```
  lib-page-chrome.js
  lib-pe3.js
  lib-detkit.js
  scene-hero.js
  viz-hero.js
  lib-archflow.js
  scene-classifier.js
  <your scene-*.js modules>
  <your ch-*.html and viz-*.js files, chapter by chapter>
  ```
- `notes/claims/page-<group>.json` ONLY for new claims you must add (derived values such as computed token
  counts, labelled `"our_reading": true` and with `loc` showing the computation). Reuse existing claim ids
  whenever the number exists in `notes/claims/*.json`.
- Atlas: `atlas/chapters/<atlas-id>.html` (an `<article data-chapter="<atlas-id>" data-chip="Short name">`
  following `atlas/chapters/reads.html`: `.ch-head` with `<span class="num"></span>` (the build numbers it),
  `.tools` with step tabs if useful, `.step-text` blocks, one `.note` saying what the stage shows and what is
  toy vs sourced, at most one `table.data` or `.stat`, one `.inscription`, and a
  `<button type="button" class="continue" data-go="x">Continue</button>` (the build rewrites its target)),
  `atlas/scenes/<atlas-id>.js` (`SCENES['<atlas-id>'] = ctx => { ... }` per the scene contract in
  page-authoring.md "Atlas mode"; reuse your `parts/scene-*.js` modules), and `atlas/dialogs/<atlas-id>.html`
  with `<dialog id="dlg-<term>">` glossary dialogs for every `inline-term` you use (unique ids; check
  `atlas/dialogs.html` for existing ones: detector, embedding, lvis, papers). Prose for a newcomer, every
  acronym expanded once.

## Rules that QA enforces or reviewers will check
- Numbers: every sourced number in prose, captions, tables and figure labels is `<span data-claim="id">value</span>`
  whose text contains the claim's value exactly; never show a claim marked `uncertain`. Charts that plot sourced
  data take it from `notes/data/*.json` or claims (copy the numbers into the JS with a comment naming the table)
  and carry a "Sourced" badge; computed figures "Computed"; toys "Toy model" + `data-toy` on the figure.
- Metric discipline: name the metric and setting next to every number (box vs mask AP, LVIS val vs minival,
  fixed AP, ODinW-13 vs ODinW-35, zero-shot vs fine-tuned). Never put unlike metrics side by side without saying so.
- Our own inference is labelled `<span class="reading">our reading</span>`; code-only values say "official code".
- Prose: intuition first, then the equation, then a variable-by-variable gloss. About 80% ASD-STE100: one topic
  per sentence, active voice, short sentences, keep articles. ASCII punctuation only in code and comments; in
  prose use commas, colons, semicolons or parentheses instead of dashes. No uppercase eyebrows, no middle-dot
  strings, no arrows appended to links, no fade-ins. One prediction prompt per chapter (`<p class="predict">`).
  Reader: basic deep learning only; CLIP and DETR/DINO are recapped in one paragraph with a link
  (`../vlm-survey/explainer.html#ch-shared-space`, `../detr-family/explainer.html#ch-denoising` etc.); two-stage
  detectors and YOLO necks are taught from scratch.
- Equations: TeX from the paper's LaTeX, built with `\htmlData{step=N, link=word}{...}` inside
  `<div class="eq" id="eq-..." data-build data-viz="viz-...">`, colors via `\htmlClass{t-blue}{...}`, followed by
  `<p class="eqno">Eq. N in <paper> (<arxiv id + version>)</p>` and a `<ul class="gloss">`. Every link word must
  light something real in the named figure (`PE.linkPulses(fig, redraw)` returns `glow(word)`).
- Figures: `<figure class="viz" id="viz-<slug>">` with a badge, a canvas (or svg) with role="img" and an
  aria-label, `.controls` of real `<button id="btn-<slug>-<state>">` elements, one reset button with
  `data-cue-reset`, a `<p class="readout" aria-live="polite">` written once per click, and a `<figcaption>`.
  Buttons SET a state (pressing twice = once). Animate every state change with `PE.animate` (400-900 ms);
  jump to the end state when `PE.instant()`. Canvas via `PE.canvas(el, aspect, draw)`; read colors with
  `DK.C()` on every draw (both themes). Layout must work at 375 px (no horizontal scroll; restack below ~620 px;
  labels at least 10 px) and at 1280/1440 px. Draggable things also move with arrow keys.
- Architecture chapters get the archflow walkthrough ("Inside <model>: forward, then backward") with shapes
  computed from the paper's config (config values as data-claims in the stage texts; code-only labelled;
  shapes we computed labelled our reading) and the backward pass showing what trains and what is frozen.
  Analysis chapters get claim cards (`<div class="claimcard">` with h4 + dl: Claim / Measured / Result /
  Would falsify).
- Scene modules shared with the atlas may use only: PE.css, instant, canvas, animate, rng, gauss, label, clamp,
  lerp, mix, dot, arrow, polyline, axes, softmax, view; DK.*; PE3 (pe3.js). `PE.canvas(...)` returns
  `{ redraw, state, ctx }` on the long page and also `{ resize, destroy }` in the atlas: call `destroy` only if it exists.
  Never nest `PE.mix` inside `PE.mix` (it returns rgb()); use `DK.mixh` for anything you mix again.

## Build and QA (run until clean; never touch the shared outputs)
```
cd ~/paper-explainers/ovd-survey
bash build.sh parts/ORDER.<group> build/private/<group>          # long page: must print PASS
bash build-atlas.sh atlas/ORDER.<group> build/private/<group>/atlas.html   # atlas: must print problems: 0
```
`atlas/ORDER.<group>` lists `overview`, `reads`, then your atlas ids. Then LOOK at your figures: run
`node ~/.claude/skills/paper-explainer/assets/vizshots.mjs <page> <outdir> <figId> <btn,btn,...> 1200` with
cwd `~/paper-explainers/.node` for each figure (dark, light, mobile), and read the PNGs. Fix overlaps, clipped
labels, unreadable text and empty states. Also check the atlas shots in `build/private/<group>/shots-atlas/`.
Review the "bare numbers" list that check-page prints and wrap or justify each.

## Report back (under 300 words)
Files written; claims added; anything in the outline you could not support; any bug you found in a shared file
(do not fix shared files yourself: `lib-*.js`, `scene-hero.js`, `scene-classifier.js`, `explainer.head.part`,
`build*.sh`, skill assets; describe the bug and your workaround). If you needed a new CSS class, put a `<style>`
block in your own part and say so.

# Architecture-walkthrough brief (shared by all chapter agents, 2026-09-30)

Project: ~/paper-explainers/vlm-survey (single-page, 3Blue1Brown-style VLM explainer; page only, no audio).
The user asked for architecture and training details in every architecture chapter: how image and text
flow through the whole model (tensor shapes), and how the gradient goes back (what is trained, what is
frozen, what the loss grades). Chapter 1 (CLIP) is done and approved as the template.

## Read first, in this order
1. ~/.claude/skills/paper-explainer/references/page-authoring.md, the bullets "Architecture walkthrough"
   and "Claim card" (the standard), plus "The 3Blue1Brown standard" and "Cue contract".
2. notes/chapter-brief.md: the original chapter rules (colors, ids, claims, ASCII, testing). They all still
   apply, EXCEPT the "Files you may write" list, which is replaced below.
3. The exemplar: parts/ch-shared-space.html, sections "Inside CLIP: one forward pass, shape by shape" and
   "Backward: where the gradient goes"; parts/viz-clipflow.js (an ArchFlow spec); parts/viz-clipgrad.js
   (a gradient toy). Match their depth, tone, honesty labels ("our reading", "code-only", "derived here").
4. parts/lib-archflow.js: the shared ArchFlow component (API in its header comment). Use it for the stepper;
   do not modify it. If it cannot express something you need, say so in your report and draw that part in
   your own small figure instead.
5. Your chapter's current part (parts/ch-<slug>.html) and its existing viz files, outline.md (incl. the
   "Corrections" section), notes/papers/<key>.md, notes/claims/<key>.json, and the primary sources
   sources/<key>/ (LaTeX in src/, paper.pdf). Architecture configs, frozen/trainable per stage, losses and
   optimizer settings come ONLY from these sources. If a value is not in the paper (e.g. an LLM's hidden
   width), either write it symbolically (d_LLM) or, if it is in a cited paper's source we have, cite that;
   never from memory.

## What to add to each of your chapters
- An `<h3>Inside <model>: forward, then backward</h3>` section inserted after the chapter's main idea and
  equation, before its closing bridge paragraph. Two-sentence setup, then a `<figure class="viz archflow"
  id="viz-<slug>flow" data-computed>` built with ArchFlow (copy the CLIP markup: stage pill row with
  forward/backward groups, optional variant row for model sizes OR training stages, canvas, prev/next/reset,
  readout, `.cf-text` stage divs with data-claim spans, figcaption with a prediction prompt).
  - Forward stages: one idea each, inputs -> tokens -> encoder(s) -> connector -> splice/sequence -> LM or
    head -> loss. Readout gives exact tensor shapes COMPUTED from the paper's config (say "per image" or
    symbolic N / L where the paper gives no batch).
  - Backward stages: gradients (with the frozen distinction: brown "gradient passes, no update" vs
    "gradient stops here"), and update (optimizer, lr, batch, steps, compute from the paper).
  - Variants: training stages are the most useful variant when a model trains in stages (e.g. which
    boxes are frozen in stage 1 vs 2); model sizes when shapes differ by size.
  - Say what the loss grades (which positions, which targets) in the loss stage text.
- A gradient toy ONLY where the gradient's shape is the insight (e.g. a zero-initialised gate, a mask,
  a per-pair loss). Simulate it in build/sim/ first; derive any gradient equation on the page and label it
  derived. Otherwise skip it: the stepper is required, the toy is not.
- For analysis/ablation chapters (ch-recipes): no stepper; instead a claim card per paper
  (`<div class="aside claimcard">`): the claim in one sentence, how it was measured (data, metric,
  baselines, what was held fixed), the result with data-claim, what would have falsified it.

## Files you may write (nothing else)
- parts/ch-<slug>.html for your chapters (edit in place; keep existing content and ids)
- parts/viz-<slug>flow.js (and parts/viz-<slug>grad.js if you add a toy)
- notes/claims/arch-<slug>.json (new claims for your chapter; ids `<paperkey>-arch-...`, same schema as
  notes/claims/clip.json; check an id does not already exist in notes/claims/*.json)
- build/fork-arch-<slug>/, build/fork-arch-<slug>.order, build/sim/arch-<slug>-*.cjs
Never edit parts/ORDER, build.sh, explainer.head.part, parts/lib-archflow.js, other chapters, or existing
claims files. Do NOT spawn sub-agents. Do not run the full build.sh without arguments.

## Test (must PASS before you return)
```
cd ~/paper-explainers/vlm-survey
printf 'lib-archflow.js\nch-<slug>.html\n<its viz files in parts/ORDER order>\nviz-<slug>flow.js\n' > build/fork-arch-<slug>.order
bash build.sh build/fork-arch-<slug>.order build/fork-arch-<slug>
mkdir -p build/fork-arch-<slug>/viz && cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/vizshots.mjs \
  ~/paper-explainers/vlm-survey/build/fork-arch-<slug>/explainer.html ~/paper-explainers/vlm-survey/build/fork-arch-<slug>/viz \
  viz-<slug>flow btn-<slug>flow-<a stage>,btn-<slug>flow-backward,... 2600
```
(The fork build contains only your chapter, so links to other chapters are dangling there; that is fine.)
In the real page, viz files are loaded after their chapter html: list new viz files in your report so the
integrator adds them to parts/ORDER right after your chapter's existing viz files.
READ the PNGs (dark, light, 375 px mobile): no overlapping labels (under-box labels must fit the box
width; use '|' to split lines), frozen boxes visibly brown, backward arrows correct.

## Return (15-25 lines)
Per chapter: sections added, figure stages + variants, where frozen/trainable came from (paper + loc),
claims added (count, file), toy added or not and why, new viz files for ORDER, QA result, and every
place you had to write a value symbolically or label it code-only/our reading.

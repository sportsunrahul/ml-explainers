# Chapter authoring brief (forks building chapters in parallel)

Write ONLY your own files: `parts/ch-<id>.html`, `parts/scene-<name>.js`, `parts/ORDER.<you>`, extra claims in
`notes/claims/extra-<you>.json`, and your private build under `build/test-<you>/`. Do NOT edit shared files:
explainer.head.part, parts/body-top.html, parts/lib-*.js|html, parts/ORDER, build.sh, atlas/*, outline.md,
other agents' parts or claims. If you need a shared change (a TOY helper, a CSS rule), put a uniquely named helper
in your own scene file or a <style> block at the top of your chapter html, and list it in your report.

Private build: `bash build.sh parts/ORDER.<you> build/test-<you>` where ORDER.<you> = the lib lines of parts/ORDER
(lib-page-chrome.html, lib-pe3.html, lib-archflow.html, lib-scenekit.js, lib-toy.js) + your chapter and scene
files. Screenshots: `cd ~/paper-explainers/.node && node sam-shots.mjs <html> <outdir> [selector] [widths]`,
`node sam-tabs.mjs <html> <outdir> <figId> <btnIds,comma> [width] [scheme]`, `node sam-eqlink.mjs <html> <outdir> <eqId>`.
Read the PNGs and fix until clean at 1280 and 375, dark and light. check-page must PASS on your private build.

Page rules (references/page-authoring.md is the authority; read it once):
- Section `<section id="ch-<id>" class="chapter">` with `<h2>` first. Act kicker `<p class="kicker">` only on the
  first chapter of an act (ch-promptable: Act II "Segment Anything (2023)"; ch-sam2: Act III "Time, concepts, depth").
- Two-sentence setup, then the figure, then prose, then equation (built with \htmlData{step=N, link=word}), gloss.
  One prediction prompt per chapter. Every link word must light something real in the figure (pe-link event on
  the figure: `ctx.article.addEventListener('pe-link', ...)`, remove it in destroy).
- Scene figures: `<figure class="viz" id="viz-<x>" data-scene="<name>" [data-toy]>` with `.stage` div, controls
  (buttons with ids `btn-<x>-...` AND data attributes the scene binds through ctx.q/ctx.qa), a reset button with
  `data-cue-reset`, `<p class="readout" aria-live="polite">`, figcaption. Tabs: `[role=tab][data-step=n]` inside a
  `.seg[role=tablist]`. Cue contract: buttons set state (idempotent), animations jump to the end under PE.instant().
  The scene contract is in parts/lib-scenekit.js and assets/atlas/shell.html: SCENES.<name> = ctx => ({ flat:true,
  step(n), replay(), resize(), destroy() }). Use only PE helpers present in BOTH pages (css, instant, onTheme,
  canvas, animate, rng, gauss, label, clamp, lerp, mix, dot, ring, arrow, box, polyline, axes) plus TOY and the
  scene helpers (mkCanvas, listen, press, badge, S3ASPECT, mid). Respect `ctx.reserve` (bottom px to keep free) and
  `narrow = W < 620`. Leave the top-right 40 px free for the badge. Every scene must also work as an atlas stage.
- ArchFlow walkthroughs (assets/archflow.js; read its header for the spec API) are long-page only, not scenes.
- Honesty: badge kinds toy / computed / sourced. Every number in prose or captions is
  `<span data-claim="id">value</span>` with the claim value verbatim. Never show `uncertain` claims as fact.
  If you need a number that has no claim, verify it in sources/ yourself and add it to notes/claims/extra-<you>.json
  with an exact loc. Our own inference is labelled "our reading".
- Masks are green cutouts (TOY.drawMask), prompts yellow, exclude clicks brown, image/features blue, loss red.
- Prose: from scratch for a reader with basic deep learning; about 80% ASD-STE100 (one topic per sentence, active
  voice, articles kept, one word per meaning); ASCII punctuation only; no em dashes; no uppercase eyebrows.
  Expand each acronym once. Link to existing explainers where useful: ../detr-family/explainer.html,
  ../vlm-survey/explainer.html, ../dino/dino-explained.html, ../jepa-survey/explainer.html.
- Report back in under 250 words: files written, claims added, what you could not do, shared changes requested.

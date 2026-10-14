# QA report: vlm-survey (page only, 2026-09-29)

Page: explainer.html (1.2 MB, self-contained; 13 chapters + family tree, summary, where to go deeper).
Source of truth: parts/ (assembled by build.sh in parts/ORDER order) -> explainer.src.html -> explainer.html.

## Checks run
- check-page.mjs on the integrated page: PASS (no console errors, no duplicate ids, no overflow at 375 / 1280 px,
  dark + light + reduced motion, every data-claim id resolves and its span contains the claim value).
- KaTeX: 211 expressions, 0 errors. ASCII punctuation: 0 smart quotes / dashes in the source.
- Claims: 728 in notes/claims.json, 445 used on the page, none of them uncertain.
- Fact checks (fresh agents, reports in notes/):
  - claims vs sources: factcheck-A.md, factcheck-B.md (0 wrong values; ~85 locs, one scrambled file, flags fixed)
  - chapter 8 (training step) on its own: factcheck-page-training-step.md (25 items, all applied)
  - page chapters 1-7: factcheck-page-A.md (24 items, all applied or re-worded against the source)
  - page chapters 9-13 + closing: factcheck-page-B.md (30 items, all applied or re-worded against the source)
- Every figure state screenshotted and read (build/fork-*/viz, build/viz) in dark, light and 375 px.

## Known limits / open items
- check-page.mjs never lists bare numbers inside data-toy / data-computed figures (captions included);
  those captions were checked by hand in the page fact-checks. Tool gap, not fixed in the skill.
- check-page validates data-claim spans present at page load only; figure states that inject claim spans later
  (viz-ablations tabs, viz-boxes states) were validated by a separate script in build/fork-fix3/.
- The family-tree arcs are editorial (stated on the page).
- No narration (page-only run, by request).

## Revision 2026-09-30: CLIP pipeline + gradient flow
- New: viz-clipflow (computed shapes, 9 stages, 4 model sizes), viz-clipgrad (toy, exact gradient), eq-clip-grad.
- check-page.mjs on the full page: PASS. Every new figure state screenshotted in dark, light, 375 px and read
  (build/fork-clipflow/viz); fixed mobile arrow/title overlap and a legend collision.
- Toy simulated first: build/sim/clipgrad.cjs.

## Revision 2026-09-30 (b): architecture walkthroughs in every architecture chapter
- Shared stepper component parts/lib-archflow.js (skill assets/archflow.js). New figures: viz-siglipflow, viz-siglipgrad,
  viz-vitflow, viz-blipflow, viz-flamingoflow, viz-flamingograd, viz-blip2flow(-s2), viz-llavaflow, viz-qwen2vlflow,
  viz-florence2flow, viz-chameleonflow, viz-qwen3vlflow; claim cards in ch-recipes. Claims in notes/claims/arch-*.json.
- Each chapter pair fact-checked by a fresh agent; all WRONG/MISLEADING/UNSUPPORTED items applied by the builders.
- Known limits: ArchFlow variants cannot change the graph (BLIP-2 uses two figures; Qwen2-VL has a second button row
  in its own code); Flamingo's interleaved xattn/LM stack is drawn as two boxes; Qwen3-VL desktop has two arrow crossings.

## Revision 2026-10-01: design pass
- New: hero animation (viz-hero, plays once, Replay button, final frame under reduced motion), page chrome
  (rail + progress, mobile bar + chapter menu, equation hover links), figures as 60rem stages, KaTeX_Main headings.
- check-page.mjs: PASS. All 33 figures re-shot at 1280 px dark after the width change (contact sheets in
  build/design/figs1280); hero final frame read at 1440 dark/light and 375 px; toc, act divider, equation hover
  and the mobile menu read.

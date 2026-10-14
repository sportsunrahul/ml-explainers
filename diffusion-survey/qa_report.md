# QA report: diffusion-survey (page only)

Status 2026-09-29: page complete (12 chapters + family tree, summary, deeper). Narration deferred at the
user's request, so no audio QA was run.

## Checks run
- check-page.mjs: PASS (dark, light, mobile 375 px, reduced motion; ids, claims, overflow, button clicks).
- Claims: 417 in notes/claims.json; two fresh-agent fact checks on claims (factcheck-A/B.md) and one on the
  finished page (factcheck-page.md: 4 MUST-FIX + 20 NIT, all applied or dispositioned below).
- Advisor review caught one number introduced by the fix pass (CFG T=128 best is 3.02, not 3.04); fixed.
- Source is ASCII-only.

## Dispositions of fact-check items not applied verbatim
- 23 (Toy badge on viz-latent shows sourced R-FID): badge now reads "Toy model + sourced numbers".
- 25 (Sohl-Dickstein "lagged far behind GANs"): reworded to "reported likelihoods rather than sample quality".
- 26 (Stable Diffusion built on LDM): reworded to the sourced statement (DiT uses SD's VAE, a fine-tuned LDM f8).
- 27 (few-step DDPM = blur): reworded to mention leftover noise with the larger variance.
- 28, 29 (toy sampler behaviour): measured in-session (node sims) and stated as toy-only ("here").

## Known gaps for the narration stage
- (Fixed 2026-10-01, see below) Equation link words now pulse their figures in viz-reverse, viz-steps,
  viz-guidance, viz-latent and viz-reflow. viz-dit has no equation that points at it.
- No scripts/ or lexicon.json yet.

## Revision 2026-10-01: design pass
Plan in notes/design.md. Pre-design source kept at build/explainer.src.pre-design.html.
- New: a computed hero (time runs left to right; a 1D three-cluster density spreads into noise under the cosine
  schedule; forward paths from the real noising chain, reverse paths from DDIM with the exact denoiser; plays once,
  Replay button, final frame under reduced motion). Axis ends at 0.97 T (abar = 0.0022) because the last cosine
  steps re-noise almost fully and only add zigzag. Caption under the hero states what is computed.
- Page chrome (assets/page-chrome.js, inlined verbatim): chapter rail with progress, mobile bar and chapter menu,
  60rem figure stage, numbered h2s, equation hover links. vlm-survey design CSS (KaTeX_Main headings, pill controls,
  sentence-case badges, act dividers). Paper list as arXiv pill links; toc extras as a list; family tree as a dated line.
- Equation links: 9 equations across 5 figures lit only the term. Each link word now lights a drawn layer
  (21 terms, each focused and shot at the pulse peak: build/design/links, links-375 light).
- Fixed: viz-steps legend clipped at 375 px (now two lines).
- check-page.mjs: PASS (dark, light, 375 px, reduced motion). 0 non-ASCII characters in the source. Bare-number
  review: 212 (was 242; the arXiv ids moved into link titles; the only new entry is the "SD3" pill label).
- Every figure re-shot at 1280 px after the width change (build/design/figs1280, sheets 1-4); hero read at 1440 dark
  and light, 1280, 375 dark and light, mid-animation, and reduced motion; act divider, family tree and mobile menu read.
- No prose claim changed, so no new fact-check agent ran. The hero caption names only the cosine schedule (already
  claimed for ch-forward) and the samplers; it shows no number.
- Shared skill changes, coordinated with the detr-family session: page-chrome.js measure rule now covers every
  `main > section` (closing sections had no measure; vlm-survey has the same gap until it recopies the file);
  page-authoring.md now requires every equation link word to light a real layer.

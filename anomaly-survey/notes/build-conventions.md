# Build conventions for chapter forks (read fully before writing)

You build one to four chapters of the long page. The standards are `~/.claude/skills/paper-explainer/references/page-authoring.md`
(read "The 3Blue1Brown standard", "Equation builds", "Structure", "Cue contract", "Design", "3D stages", "Math",
"Honesty"), `notes/design.md`, and the worked example already on the page: `parts/ch-problem.html` + `parts/viz-normal-cloud.js`.

## Files
- You own ONLY: `parts/ch-<id>.html` for your chapters, `parts/viz-<slug>*.js`, and (if you need a helper) `parts/lib-<your-chapter>.js`.
- Never edit shared files: `explainer.head.part`, `parts/body-top.html`, `parts/lib-toy.js`, `parts/lib-pe3.js`,
  `parts/lib-archflow.js`, `parts/lib-page-chrome.js`, `parts/ORDER`, `build.sh`, `outline.md`, `state.json`, `notes/claims/*`.
  If you need a change there (a TOY helper, a CSS rule), put it in your own lib file and say so in your report.
- Each `.js` part is an HTML fragment: `<script>(() => { ... })();</script>` (optionally a `<style>` block first). Use an IIFE;
  no globals except a lib you own (name it after your chapter, e.g. `PATCHCORE_KIT`).

## Private build and QA (never write explainer.html in the project root)
```bash
cd ~/paper-explainers/anomaly-survey
printf '%s\n' lib-page-chrome.js lib-toy.js lib-pe3.js lib-archflow.js <your parts in order> > parts/ORDER.<fork>
bash build.sh parts/ORDER.<fork> build/test-<fork>          # merges claims, prerenders KaTeX, runs check-page
cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/vizshots.mjs \
  ~/paper-explainers/anomaly-survey/build/test-<fork>/explainer.html ~/paper-explainers/anomaly-survey/build/test-<fork>/figs <figId> <btnIds,...> 900
```
check-page must PASS. Read the screenshots of every figure (1280 dark, light, 375 mobile) and fix what looks off: overlapping
labels, clipped text, empty states, unreadable 6 px labels on phones. Review the bare-number list it prints.

## Page structure
- Act dividers: `<p class="kicker"><span class="act-n">Act II</span>Remember normal</p>` at the top of the FIRST chapter file of
  each act: ch-spade (Act II, "Remember normal"), ch-fastflow (Act III, "Model the density, or fake the defect"), ch-efficientad
  (Act IV, "Fast, and one model for all"), ch-winclip (Act V, "No training images"), ch-speed (Act VI, "Choosing a method").
- `<section id="ch-<id>" class="chapter">`, first child `<h2>` with the title from `parts/body-top.html`'s toc.
- Two sentences of setup, then the figure, then the idea, the equation (with the paper's equation number and version:
  "Eq. 3 in PaDiM v1"), a gloss list, then depth. One prediction prompt per chapter, answered by the figure.
- Method chapters (SPADE, PaDiM, PatchCore, FastFlow, DRAEM, EfficientAD, UniAD, WinCLIP, AnomalyCLIP) get
  `<h3>Inside <name>: forward, then backward</h3>` with an `archflow.js` stepper (read the header of `parts/lib-archflow.js`;
  a worked spec: `~/paper-explainers/jepa-survey/parts/viz-ijflow.js`). Shapes are COMPUTED from the config claims; frozen boxes
  brown, trainable boxes red outline; methods that train nothing say so in the backward stage ("no gradient anywhere: fitting
  stores statistics"). Benchmark/analysis chapters get a `.claim-card` (`<div class="claim-card"><dl><dt>Claim</dt><dd>..</dd>
  <dt>How measured</dt>..<dt>Result</dt>..<dt>What would falsify it</dt>..</dl></div>`).
- Ids: figure `viz-<slug>`, canvas `cv-<slug>`, buttons `btn-<slug>-<state>`, all unique on the page. Buttons SET a state
  (pressing twice = once); one `data-cue-reset` button per figure; animations use `PE.animate` and jump to the end under
  `PE.instant()`; an `aria-live` `.readout` written once per click; canvases focusable, draggables move with arrow keys.
- Structure each figure as a pure `draw(ctx, W, H, state)` plus thin DOM wiring, so the atlas page can reuse the drawing.

## Content rules
- Reader: basic deep learning, knows DINO and CLIP. Introduce every other term where first used.
- Prose at about 80% of Simplified Technical English: one topic per sentence, active voice, keep articles, short paragraphs.
  ASCII punctuation only (no em or en dashes, no curly quotes) in prose and code. No company or product names beyond the papers.
- Every sourced number is `<span data-claim="<id>">value</span>`; the id must exist in `notes/claims/*.json` and the visible
  text must contain the claim's `value` verbatim (find ids with `grep -h '"id"' notes/claims/<file>.json | grep <word>`).
  Claims with `"uncertain": true` are not shown as fact. `our_reading` values (computed shapes, sums) are allowed but label them
  "our computation" in the text or caption. Never type a number from memory.
- Toy figures carry `<span class="badge toy">Toy model</span>` and `data-toy` on the figure; sourced charts `badge sourced`;
  figures that run a paper's algorithm or config `badge computed`.
- Colors: blue normal, red anomaly/score/heat, yellow the sample under inspection, green learned, brown frozen/stored. Heat maps
  via `TOY.heatBitmap` (transparent to red); camera views via `TOY.draw` (viewfinder corners). The toy world (`parts/lib-toy.js`:
  mesh + cap, defects scratch/hole/stain/missing, 3-d patch features, `bank`, `nnScores`, `toPixels`) is the shared toy; extend it
  in your own lib, never fork its numbers. Simulate any toy in node first (`build/toy-*.mjs` shows how) and state in a code
  comment the numbers you checked.
- Apply the extraction agents' outline feedback (section 10 of each `notes/papers/*.md`; summary in `notes/resume-notes.md`).
  Where a paper and its official code disagree (PatchCore Eq. 7), say so on the page.

## Report (under 250 words)
Files written, the ORDER lines for your chapters in order, check-page result, figures screenshotted, claims used, and any
shared-file change you need.

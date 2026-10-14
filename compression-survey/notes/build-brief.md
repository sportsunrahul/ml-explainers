# Build brief (read fully before writing anything)

Project: `~/paper-explainers/compression-survey/`. You build one or two chapters of an intuition-first,
interactive explainer page (`explainer.html`) and one or more stages of a companion one-screen tour
(`atlas.html`). The reader knows basic deep learning only and deploys detection and segmentation models on
GPUs. The bar is 3Blue1Brown: the visual carries the idea, prose is the depth layer.

Read first, in this order:
1. `notes/design.md` (colors, type, principles), `outline.md` (your chapter's section in full, including the
   corrections folded into it), `notes/understanding.md`.
2. The reference implementation, and copy its patterns exactly:
   - `parts/ch-cost.html` (a finished chapter: kicker, section, figure, equation build, gloss, prediction
     prompt, "On a detector" aside, the inline mount script).
   - `parts/kit/00-base.js` (KIT helpers), `parts/kit/20-roofline.js` (a figure kit), `parts/head.html`
     (CSS classes and the `PE` helper API: css, instant, onTheme, canvas, animate, rng, gauss, label, clamp,
     lerp, dot, ring, arrow, box, polyline, axes (logx/logy), mix, view, linkPulses, softmax).
   - `parts/lib-archflow.html` header (the ArchFlow spec) if your chapter has an architecture walkthrough;
     `parts/lib-pe3.html` header (the PE3 3D API) if your chapter has a 3D figure.
   - Atlas: `atlas/parts/cost.article.html` and `atlas/parts/cost.scene.js` (a finished stage),
     `atlas/dialogs-common.html`, and `~/.claude/skills/paper-explainer/references/page-authoring.md`
     sections "Atlas mode", "3D stages", "Equation builds", "Honesty".
3. Your papers' notes `notes/papers/<name>.md` and claims `notes/claims/*.json` (search them with grep or a
   small script; never invent a number). The fact-check logs `notes/factcheck-claims-*.md` list wording traps.

## What you write (only these files; never edit anyone else's)
- `parts/ch-<slug>.html` per chapter. If your chapter opens an act, put the act kicker at its top
  (`<p class="kicker"><span class="act-n">Act N</span>Title</p>`), exactly as `ch-cost.html` does.
- `parts/kit/<NN>-<slug>.js`: your figure kits, in your assigned number range. Kits draw from a state object,
  know nothing about buttons, and accept `opts.aspect` (number or function of width), `opts.fig` (for
  `PE.linkPulses`) and `opts.bottomPad` (pixels left free at the bottom for the atlas readout). Return at least
  `{ set, state, redraw, resize, destroy }`. Put readout-text builders in the kit too (`KIT.<name>Text`), so
  both pages print the same sentence.
- `atlas/parts/<id>.article.html`, `atlas/parts/<id>.scene.js`, `atlas/parts/<id>.dialogs.html` for your atlas
  ids. Articles use `{{NUM}}` and end with `{{CONTINUE}}`; the assembler fills both.
- Notes of node simulations: `notes/sim-<slug>.txt`.

## Invariants (the page check and the reviewer enforce these)
- ASCII only in every file you write: no em or en dashes, no curly quotes, no Unicode symbols in source. In JS,
  write symbols as escapes (`'×'`). No tab characters in kits.
- Ids: chapter `ch-<slug>`; figures `viz-<name>`; buttons `btn-<name>-<what>`; equations `eq-<name>`; canvases
  `<name>-canvas`; readouts `<name>-readout`. Prefix every id with your figure's name: duplicate ids anywhere on
  the page fail the check.
- Each chapter: `<section id="ch-<slug>" class="chapter">` whose FIRST child is its `<h2>`; a two-sentence setup,
  then the main figure, then prose; at least one display equation built term by term; a variable-by-variable
  gloss; one explicit prediction prompt ("Make a guess before you press ..."); an `.aside.detector` "On a
  detector" note as the last element before the script; cross-links to other chapters by `#ch-<slug>`.
- Each figure: `<figure class="viz" id="viz-...">` with a badge (`<span class="badge toy">Toy model</span>`,
  `badge computed` "Computed", or `badge sourced` "Sourced") AND the matching attribute on the figure
  (`data-toy`, or `data-computed` for computed toys; a sourced chart needs neither but every number in it is a
  data-claim). Controls are real `<button>`s with ids; state buttons SET a state (pressing twice = once; two
  buttons for before/after, never a toggle); a reset button with `data-cue-reset` and class `reset`; an
  `aria-live="polite"` readout written once per click, never per frame; a `<figcaption>` that says what is toy,
  computed or sourced; `role="img"` and an `aria-label` on canvases.
- Every state change animates (`PE.animate` or `KIT.tween`, 400-900 ms) and jumps to the end state when
  `PE.instant()` is true. No autoplay except a figure's own replay button.
- Equations: TeX in `<div class="eq" id="eq-..." data-build data-viz="viz-..."><div class="tex tex-display">`
  with each term wrapped `\htmlData{step=N, link=word}{...}` and colored with `\htmlClass{t-blue}{...}` etc.
  Every link word must light something real in the figure (read `PE.linkPulses(fig, redraw)` in the kit and
  glow the named layer, or put `data-link="word"` on an HTML element in the figure). Inline math:
  `<span class="tex">...</span>`. Below each display equation: `<p class="eqno">Eq. N in <paper>, arXiv
  id+version</p>`. Quote equations from the notes (which are verbatim from the LaTeX); if you rewrite one in
  page notation, say "our notation".
- Numbers: every sourced number in prose, captions, tables or readouts is wrapped
  `<span data-claim="<id>">...</span>`, the id exists in `notes/claims/*.json`, and the claim's `value` string
  appears verbatim inside the span. Never show a claim marked `"uncertain": true` as fact (a plot reading may
  appear as "about N, read off the figure" inside `<span data-approx>`). Relative-percent deltas (Wu) and
  absolute points (Nagel, BRECQ, van Baalen, Mishra) are never mixed in one column. Every table row names
  model, backbone, dataset, metric and toolchain where the source gives them. Labels: "our reading" for our
  inference, "code-only" for repo-only facts, "our transcription" for equations the paper gives in prose.
- Toys that train, optimise or search (STE descent, GPTQ, AWQ alpha search, KL calibration, CLE, IMP) run once
  in node first; write the settings and the numbers the reader will see to `notes/sim-<slug>.txt` and make sure
  the visible effect is real at those settings (a toy that shows nothing is a bug).
- Colors only via the five CSS variables, with the page key's meanings: blue = full precision / teacher;
  yellow = integer grid / quantized / student; green = kept or protected / fused; red = error, loss,
  gradient; brown = removed or frozen. Canvas code reads them with `PE.css('--blue')` at draw time.
- Layout: canvases size from width with an aspect function; give a narrow layout below 560 px (labels no
  smaller than 10 px); no horizontal overflow at 375 px. 3D (pe3): pitch above about .5 rad for surfaces; label
  the three or four things that matter where they are drawn.
- Prose: plain technical English for a newcomer. One idea per sentence, active voice, sentences under about
  25 words, every term defined at first use, every acronym expanded once. No filler, no "In this chapter".
  Intuition first, then the equation, then the gloss.

## Testing (do all of it before you report)
1. Long page, private build: write `build/test-<group>/ORDER` listing `ch-cost.html` and your chapter files,
   then run `bash build.sh build/test-<group>/ORDER build/test-<group>` from the project root. It must end in
   `PASS`. Review the "bare numbers" list it prints: each must be a shape, a bit-width, a chapter reference, or
   wrapped in a data-claim.
2. Screenshots: `cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/vizshots.mjs
   <abs path>/build/test-<group>/explainer.html <abs path>/build/test-<group>/figs <viz-id> <btn,btn,...>`.
   Read the PNGs (dark, light, mobile). Fix overlaps, clipped labels, tiny text, empty states.
3. Equation links: hover each `[data-link]` term once in a quick Playwright script (or reason it through) and
   confirm something lights.
4. Atlas, private build: `node atlas/assemble.mjs <abs path>/build/test-<group> <id> [<id>...]`, then
   `cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/atlas/test.mjs
   <abs>/build/test-<group>/atlas.html <abs>/build/test-<group>/atlas-shots`. It must report 0 problems. Read
   the desktop and mobile shots. Atlas 2D scenes leave the bottom 130 px free on desktop
   (`bottomPad: innerWidth > 860 ? 130 : 0`). Then pull every number out of your article text with a regex and
   check each against `notes/claims/*.json`; fix or label any that are not there.
5. Never edit `parts/ORDER`, `parts/head.html`, `parts/body-top.html`, other builders' files, or the skill's
   assets. If you need a new shared helper, put it in your own kit file under a `KIT.` name. If you find a bug
   in a shared file, describe it in your report instead.

Report back in under 300 words: files written, figures and what each computes, equation ids, claims used (count),
anything you could not source, and any problem in the outline or shared files.

# Build brief (read fully before writing anything)

Project: `~/paper-explainers/contrastive-survey/`. You build one or two chapters of an intuition-first,
interactive explainer page (`explainer.html`) on contrastive learning. Long page only: no atlas, no narration.
The reader knows basic deep learning only (a CNN, softmax, cross-entropy, SGD). The bar is 3Blue1Brown: the
visual carries the idea, prose is the depth layer.

Read first, in this order:
1. `notes/design.md`, `outline.md` (your chapter's section in full, including corrections folded into it),
   `notes/understanding.md`.
2. The reference implementation, and copy its patterns exactly:
   - `parts/ch-compare.html` (a finished chapter: act kicker, section, setup, figure, prediction prompt,
     equation build, eqno, gloss, "In practice" aside, the inline mount script).
   - `parts/kit/00-base.js` (KIT helpers), `parts/kit/05-circle.js` (NT-Xent on 2D unit vectors: reuse it, do
     not rewrite it), `parts/kit/20-pullpush.js` (a figure kit: pure math first, then the drawing, then a
     `KIT.<name>Text` readout builder), `parts/head.html` (CSS classes and the `PE` helper API: css, instant,
     onTheme, canvas, animate, rng, gauss, label, clamp, lerp, dot, ring, arrow, box, polyline, axes (logx/logy),
     mix, view, linkPulses, softmax).
   - `parts/lib-archflow.html` header (the ArchFlow spec) if your chapter has an architecture walkthrough;
     `parts/lib-pe3.html` header (the PE3 3D API) if your chapter has a 3D figure.
   - `~/.claude/skills/paper-explainer/references/page-authoring.md` sections "Equation builds", "Structure"
     (architecture walkthrough, claim card, interlude), "3D stages", "Honesty".
   - Reference figures in sibling projects (read, do not copy blindly): `../vlm-survey/parts/` for CLIP
     (`viz-clipflow`, `viz-clipgrad`), `../jepa-survey/parts/` for pe3 collapse figures (`viz-collapse.js`).
3. Your papers' notes `notes/papers/<name>.md` and claims `notes/claims/*.json` (search them with grep or a small
   script; never invent a number). The fact-check logs `notes/factcheck-claims-{AE,B,CD}.md` list wording traps (claim fixes are already applied; their notes-level problems are not, so trust the logs over the notes where they differ).

## Voice: teach the concept (the user's explicit request, 2026-10-13)
The user asked that every chapter read "as if you are teaching me the concept". `parts/ch-compare.html` is the
model; match its shape and voice:
- Open with the question the learner would ask, in one or two sentences, and say what the chapter will answer.
- Build the idea in small `<h3>` steps (for example "The game", "First attempt", "The fix", "The same idea as an
  equation", "Where this came from", "Where we go next"). Each step adds one idea and says why it is needed.
- Second person and plain words: "you", "imagine", "try". One everyday analogy per chapter where it helps
  (the "student who always says same" for collapse; the police lineup for InfoNCE). Do not stack analogies.
- Motivate before you formalize: the reader sees the problem in the figure before the fix is named.
- After each display equation, a worked example with small numbers the reader can check by hand.
- Name the likely confusion and resolve it ("You might expect X. In fact Y, because ...").
- At least one `<details class="check">` self-check with the answer inside, per chapter.
- Papers come second: the mechanism is taught first; who did it and when goes in a short "Where this came
  from" step. Sourced numbers support the lesson; they are not the lesson.
- End with `<div class="recap"><h4>You should now be able to explain</h4><ul>` 2-4 items `</ul></div>`, then
  the "In practice" aside last.
- Keep sentences short; define every term at first use; never assume the reader read a paper.

## What you write (only these files; never edit anyone else's)
- `parts/ch-<slug>.html` per chapter. If your chapter opens an act, put the act kicker at its top, exactly as
  `ch-compare.html` does.
- `parts/kit/<NN>-<slug>.js`: your figure kits, in your assigned number range. Kits draw from a state object,
  know nothing about buttons, accept `opts.fig` (for `PE.linkPulses`), and return at least
  `{ set, state, redraw, resize, destroy }`. Put readout-text builders in the kit (`KIT.<name>Text`).
- Extra claims you need (for example a derived number you computed from sourced ones) go in
  `notes/claims/page-<slug>.json`, with `loc` saying how it was derived.
- Simulations: `build/sim/<slug>.js` and their output `notes/sim-<slug>.txt`.

## Invariants (the page check and the reviewer enforce these)
- ASCII only in every file you write: no em or en dashes, no curly quotes, no Unicode symbols in source. In JS,
  write symbols as escapes (`'τ'`). No tab characters in kits.
- Ids: chapter `ch-<slug>`; figures `viz-<name>`; buttons `btn-<name>-<what>`; equations `eq-<name>`; canvases
  `<name>-canvas`; readouts `<name>-readout`. Prefix every id with your figure's name: duplicate ids fail.
- Each chapter: `<section id="ch-<slug>" class="chapter">` whose FIRST child is its `<h2>`; a two-sentence
  setup, then the main figure, then prose; at least one display equation built term by term; a
  variable-by-variable gloss; one explicit prediction prompt ("Make a guess before you press ..."); a `.recap` block, then an
  `.aside.practice` "In practice" note as the last element before the script; cross-links by `#ch-<slug>`.
- Each figure: `<figure class="viz" id="viz-...">` with a badge (`badge toy` "Toy model", `badge computed`
  "Computed", or `badge sourced` "Sourced") AND the matching attribute (`data-toy`, or `data-computed`; a sourced
  chart needs neither but every number in it is a data-claim). Controls are real `<button>`s with ids; state
  buttons SET a state (pressing twice = once; two buttons for before/after, never a toggle); a reset button with
  `data-cue-reset` and class `reset`; an `aria-live="polite"` readout written once per click, never per frame; a
  `<figcaption>` that says what is toy, computed or sourced; `role="img"` and an `aria-label` on canvases.
- Every state change animates (`PE.animate` or `KIT.tween`, 400-3000 ms) and jumps to the end state when
  `PE.instant()` is true. No autoplay except the hero.
- Equations: TeX in `<div class="eq" id="eq-..." data-build data-viz="viz-..."><div class="tex tex-display">`
  with each term wrapped `\htmlData{step=N, link=word}{...}` and colored with `\htmlClass{t-blue}{...}` etc.
  Every link word must light something real in the figure (`PE.linkPulses(fig, redraw)` in the kit, glow the
  named layer). Inline math: `<span class="tex">...</span>`. Below each display equation:
  `<p class="eqno">Eq. N in <paper>, arXiv id+version</p>`. Quote equations from the notes (verbatim from the
  LaTeX); if you rewrite one in page notation, say "our notation"; a derivation of ours says "derived here".
- Numbers: every sourced number in prose, captions, tables or readouts is wrapped
  `<span data-claim="<id>">...</span>`, the id exists in `notes/claims/*.json`, and the claim's `value` string
  appears verbatim inside the span. Never show a claim marked `"uncertain": true` as fact (a plot reading may
  appear as "about N, read off the figure" inside `<span data-approx>`). ImageNet numbers always name the
  protocol (linear probe, fine-tune, kNN, zero-shot), backbone and epochs where the source gives them, and
  numbers from different papers' own tables are never presented as one controlled comparison.
- Toys that train or optimise run once in node first; write the settings and the numbers the reader will see
  to `notes/sim-<slug>.txt` and make sure the visible effect is real at those settings.
- Colors only via the five CSS variables, with the page key's meanings: blue = anchor / view A / image side;
  yellow = positive / view B / caption; green = agreement, the pull, alignment; red = negatives, the push, the
  loss, the gradient; brown = no gradient (momentum encoder, bank, queue, stop-gradient). Canvas code reads them
  with `PE.css('--blue')` at draw time.
- Layout: canvases size from width with an aspect function; give a narrow layout below 560 px (labels no smaller
  than 10 px); no horizontal overflow at 375 px. 3D (pe3): label the three or four things that matter.
- Prose: plain technical English for a newcomer. One idea per sentence, active voice, sentences under about 25
  words, every term defined at first use, every acronym expanded once. No filler, no "In this chapter".
  Intuition first, then the equation, then the gloss. Generic wording: no company names or internals.

## Testing (do all of it before you report)
1. Private build: write `build/test-<group>/ORDER` listing `ch-compare.html` and your chapter files, then run
   `bash build.sh build/test-<group>/ORDER build/test-<group>` from the project root. It must end in `PASS`.
   Review the "bare numbers" list it prints: each must be a shape, a chapter reference, or wrapped in a claim.
2. Screenshots: `cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/vizshots.mjs
   <abs path>/build/test-<group>/explainer.html <abs path>/build/test-<group>/figs <viz-id> <btn,btn,...>`.
   Read the PNGs (dark, light, mobile). Fix overlaps, clipped labels, tiny text, empty states. Animated figures
   are shot mid-animation; also check the end state (reduced motion jumps to it).
3. Equation links: confirm each `[data-link]` term lights something (a quick Playwright script or reasoning).
4. Never edit `parts/ORDER`, `parts/head.html`, `parts/body-top.html`, other builders' files, or the skill's
   assets. If you need a new shared helper, put it in your own kit file under a `KIT.` name. If you find a bug
   in a shared file, describe it in your report instead.

Report back in under 300 words: files written, figures and what each computes, equation ids, claims used
(count), anything you could not source, and any problem in the outline or shared files.

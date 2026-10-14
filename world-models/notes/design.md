# Design pass: World models and video generation, explained in 3D (2026-10-01)

The page keeps its identity: Archivo + Petrona, its time/act/real/imag tokens (purple time and model,
orange action, green real, pink imagined), the 3D template and its drag-to-rotate stages. No stage on
this page auto-rotates, and the hero does not either.

## Changes
- Hero (new, illustrative): a rollout. Three real frames of the f-loop platform game (green borders),
  then four steps. In each step an orange action cube ("right", "jump", "right", "right") drops into a
  purple world-model block. The block reads the previous frame and emits the next frame (pink border),
  which flies into its slot and feeds the next step. The character walks right, jumps onto the ledge
  and walks on. Labels: agent's actions, world model, real frames, imagined frames, time. Plays once
  (6 s), ends static, Replay button, final frame under reduced motion. Below canvas aspect 1.6 the row
  tightens and the 3D labels grow by 1.45 so they stay legible at 375 px.
- Section bar: current-section highlight, reading-progress line, and a Sections menu whenever the links
  do not fit (here at every width: nine links; the old bar cut "Key models" off at 1440).
- Prose alignment: `.wrap.prose` sections (The idea, Robotics, Pixels or latents, footer) were centered
  at 700px while the others sat at the wrap's left edge; all prose now shares one left edge.
- Footer sources: one list item per reference instead of middle-dot joined runs.
- The KaTeX stylesheet link that `common.apply` adds is removed: the page has no math.

## Equation transcription
None. The page has no `.eq` blocks.

## Equation links
None.

## Not changed
No prose, no number, no figure behavior. The hero's frame drawer copies the f-loop game's colors
instead of hoisting `draw()` out of `buildLoop`, so the existing figure code stays untouched.

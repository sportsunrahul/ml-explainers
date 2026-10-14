# Design plan: JEPA, a literature survey (2026-10-08)

Subject: learning by predicting in embedding space. The whole field is one picture: a thing you can see
(blue), a thing you must guess (yellow), a guess made not in pixels but in a space of meaning (green),
and the distance between guess and truth (red). Audience: someone with basic deep learning only, who wants
to understand this deeply, by playing with it. Primary job: make each mechanism click before the equation.

## Tokens
Color (3Blue1Brown palette from the skill, dark first; one meaning each, whole page):
- bg #16181d, fg #e8e6e3, muted #9aa0a6, rule #2a2d34, panel #1d2026, panel2 #23272e
- blue #58c4dd = context, what the model sees; yellow #f4d35e = target, what it must predict
- green #83c167 = predictor and its output; red #fc6255 = loss, energy, gradient
- brown #c8906d = frozen or discarded (EMA target encoder, pixels thrown away, stop-gradient)
Light theme keeps the roles (values from the skill's base tokens).
3D figures use the same five colors plus a depth cue (fainter with distance); no lighting, no gradients.

Type: display = KaTeX_Main (embedded with the math, so headings share the equations' face); body =
Charter 1.125rem/1.7; controls, readouts, captions = system-ui. Scale: h1 clamp(2.5rem, 5.6vw, 4.2rem);
act titles 2rem; h2 2.1rem; h3 1.3rem; readouts .92rem. Prose measure 40rem; figures and display
equations break out to the 60rem stage. Everything left-aligned to one edge.

Layout (page-chrome.js): a fixed left rail with the five acts and reading progress from 1180 px up;
below that a sticky bar with the current chapter and a menu.

```
>= 1180 px                                            < 1180 px
+-----------+-----------------------------------+     +-----------------------------+
| I Labels  | 3  I-JEPA                         |     | 3  I-JEPA          [menu]   | sticky
| o 1 blank | two sentences of setup            |     +-----------------------------+
| o 2 energy| +-------------------------------+ |     | two sentences               |
| II Real   | | figure: the stage (60rem)     | |     | +-------------------------+ |
| * 3 IJEPA | |  3D or canvas, controls below | |     | | figure, full width      | |
| o 4 ...   | +-------------------------------+ |     | +-------------------------+ |
|           | prose (40rem) ... equation ...    |     | prose ...                   |
+-----------+-----------------------------------+     +-----------------------------+
```

## The hero
A 3D embedding space (pe3). An image's patches rise as a cloud of blue points; a few turn yellow and
drop out (masked); a green arrow from the blue cloud's centre lands near where the yellow points were;
a red segment measures the miss, shrinks over a few steps, and the scene settles into a static diagram
with the four labels. Six seconds, plays once, Replay button, final frame under reduced motion.
It is the whole survey in one motion: see, hide, predict in embedding space, measure the miss.

## Principles
1. The figure is the stage; prose is the depth layer. Each chapter opens with two sentences, then the
   figure, then the idea, then the equation, then the gloss.
2. One prediction prompt per chapter, answered by the figure.
3. Depth only where depth is the idea: the embedding cloud (collapse, isotropy), the energy surface,
   the video cube, the latent planner, the hierarchy. Everything else is 2D.
4. Equations point: every term lights the part of the figure it names.
5. Honesty on the figure: a "Toy model" badge where the data is synthetic, "Computed" where the figure
   runs the paper's own algorithm or config, "Sourced" where it plots the paper's numbers.

## Rejected defaults
- Uppercase eyebrows, middle-dot meta strings, arrows on links, gradient washes, shadows on cards.
- Fade-in per section. The hero is the one motion moment; every other animation answers a click or a cue.
- Numbers on things that are not a sequence: chapters are numbered (they are a sequence); figures are not.
- three.js: a 600 KB dependency and a WebGL context per figure; the figures here are point clouds, lines
  and surfaces, which a 2D-canvas projector draws well, offline, with no console noise in QA.
- A dark "cinema" treatment with glowing lines: the figures are instruments, not posters.

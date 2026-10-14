# Design plan: finding defects you have never seen (2026-10-09)

Subject: machine-vision inspection. A camera above a line looks at part after part; almost every part is
good; the rare bad one carries a defect nobody photographed before. Every method on the page answers one
question: what is the model of "normal", and how far is this patch from it? Audience: basic deep
learning, knows DINO and CLIP, wants depth. Primary job: make each family's model of normal tangible
before its equation, then let the reader compare the families on score and speed.

## Tokens

Color (3Blue1Brown palette from the skill, dark first; one meaning each, whole page):
- bg #16181d, fg #e8e6e3, muted #9aa0a6, rule #2a2d34, panel #1d2026, panel2 #23272e
- blue #58c4dd = normal: good parts, normal features, the model of normal, "pass"
- red #fc6255 = anomaly: defects, anomaly scores and heat maps, distances that flag, "reject"
- yellow #f4d35e = the test sample under inspection (the patch or part being scored)
- green #83c167 = a learned component: trained weights, the student, learned prompts, a flow
- brown #c8906d = frozen or stored: pretrained backbones, memory banks, frozen CLIP
Light theme keeps the roles (values from the skill's base tokens).
Heat maps are a single-hue ramp: transparent to red, never a rainbow (jet) map. A rainbow map invents
boundaries the score does not have, and red already means anomaly on this page.

Type: display = KaTeX_Main (embedded with the math, so headings share the equations' face); body =
Charter 1.125rem/1.7; controls, readouts, captions = system-ui with tabular numbers. Scale: h1
clamp(2.5rem, 5.6vw, 4.2rem); act titles 2rem; h2 2.1rem; h3 1.3rem; readouts .92rem.
Prose measure 40rem; figures and display equations break out to the 60rem stage; all left-aligned.

## One toy world, used everywhere

All toy figures share one procedural kit (`parts/lib-toy.js`): two products, the way MVTec AD splits
textures and objects. A woven mesh (texture) and a round machined cap with six bolt holes and a
stamped ring (object). Defects: scratch, hole, stain, bent thread, and one logical defect (a missing
bolt). The kit also has a toy feature extractor: each 16x16 patch maps to a 3-vector of patch statistics
(mean brightness, oriented-edge balance, local contrast). So the clouds in ch-problem, ch-padim and
ch-patchcore are computed from the toy images, not random dots; a scratch lands far from the cloud
because its edges point the wrong way. Captions say "toy model, features computed from the toy image".

## Layout

page-chrome.js: a fixed left rail with six acts and reading progress from 1180 px up; below that a
sticky bar with the current chapter and a menu.

```
>= 1180 px                                            < 1180 px
+-----------+-----------------------------------+     +-----------------------------+
| I Problem | 6  PatchCore                      |     | 6  PatchCore       [menu]   | sticky
| o 1 ...   | two sentences of setup            |     +-----------------------------+
| II Memory | +-------------------------------+ |     | two sentences               |
| * 6 Patch | | figure: the stage (60rem)     | |     | +-------------------------+ |
| III ...   | |  [image panel] [feature space]| |     | | image panel             | |
|           | |  controls, readout, badge     | |     | | feature space below     | |
|           | +-------------------------------+ |     | +-------------------------+ |
|           | prose (40rem) ... equation ...    |     | prose ...                   |
+-----------+-----------------------------------+     +-----------------------------+
```

Image panels (what the camera sees) get thin viewfinder corner marks; charts and feature spaces do not.
That is the one structural device, and it means "this is the camera's view".

## The hero

Canvas 2D, left to right. A line of parts slides under a circular inspection light. Each good part
sheds a few patch features that drift right and settle into a blue cloud in a small feature-space panel.
The fourth part carries a scratch. Its scratched patch turns yellow, its feature flies right and lands
far outside the cloud; a red segment grows from the nearest blue point to it; the part gets a red heat
map over the scratch and a "reject" tag. Ends static with four labels: normal features, test patch,
distance = anomaly score, anomaly map. About six seconds, plays once, Replay button `btn-hero-replay`,
final frame under reduced motion. It is the whole survey in one motion: remember normal, measure the
distance, map it back onto the part.

## Principles
1. The figure is the stage; prose is the depth layer. Each chapter: two sentences, the figure, the idea,
   the equation, the gloss.
2. One toy world. The same mesh and cap, the same scratch, across every toy figure, so the reader
   compares methods on the same defect instead of re-learning a picture each chapter.
3. Every score shows both levels: an image verdict (pass or reject) and a pixel map.
4. Equations point: every term lights the part of the figure it names.
5. Honesty on the figure: "Toy model" where the data is synthetic, "Computed" where the figure runs the
   paper's own algorithm or config, "Sourced" where it plots the paper's numbers.

## Rejected defaults
- A rainbow heat map; a product photo; a stock "AI eye" hero; a big-number stat strip.
- Uppercase "PASS / FAIL" stamps and eyebrow labels; middle-dot meta strings; arrows on links; gradient
  washes; shadows on cards; fade-in per section. The hero is the one motion moment.
- Numbers on things that are not a sequence: chapters and pipeline stages are numbered; figures are not.
- three.js or WebGL: the 3D figures are a bell, a cloud and a warped surface, which pe3.js draws offline.
- Monospace data labels: readouts use system-ui with tabular numbers.

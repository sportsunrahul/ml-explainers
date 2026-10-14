# Design plan: Segment Anything, the journey (2026-10-09)

Subject: cutting objects out of pictures. The whole field is one gesture: you point at something (a
click, a box, a word), and a shape lifts off the image. Audience: an engineer on an industrial
inspection platform with basic deep learning, studying from scratch. Primary job: make the promptable
mask click, then show each later paper as one change to that gesture.

## Tokens
Series palette (3Blue1Brown roles, kept from the earlier explainers so the library reads as one set):
- bg #16181d, fg #e8e6e3, muted #9aa0a6, rule #2a2d34, panel #1d2026, panel2 #23272e
- blue #58c4dd = the image and image features (encoder, embeddings, memory)
- yellow #f4d35e = prompts: points, boxes, text, exemplars (positive clicks are yellow dots)
- green #83c167 = predicted masks and model outputs
- red #fc6255 = loss, error regions, the gradient
- brown #c8906d = frozen, discarded, background, negative clicks
Light theme keeps the roles (skill base tokens).

Type: headings KaTeX_Main (embedded with the math CSS, zero bytes, shared with the series); body Charter
1.125rem/1.7; controls, readouts, captions system-ui. Scale as in the JEPA page.

## The signature: the cutout
Every mask on the page is drawn the same way: a translucent green fill, a crisp 2px outline, and the
cut piece lifted 3px up and left over a bg-colored gap, so it reads as a shape peeled off the picture.
Prompts sit on top: yellow dot = include, brown ring = exclude, yellow dashed box = box prompt.
One drawn toy world (a desk scene with a mug on a saucer, a lamp, a plant, a notebook) is reused by every
figure, so the reader learns its objects once; its parts form a real hierarchy (logo < mug < mug and
saucer), which is what makes ambiguity visible.

## Layout
page-chrome.js: left rail >= 1180 px, sticky chapter bar below. Prose 40rem, figure stage 60rem,
left-aligned. Atlas: one full-viewport stage, panel right, bottom sheet on phones.

```
+-----------+--------------------------------------+
| I Before  | 3  Any prompt, a valid mask           |
| o 1 masks | two sentences                         |
| o 2 click | +----------------------------------+  |
| II SAM    | | toy desk scene; click -> 3 cutouts|  |
| * 3 prompt| | lift off with score bars          |  |
|           | +----------------------------------+  |
|           | [one output] [three outputs] [reset]  |
|           | readout line                          |
+-----------+--------------------------------------+
```

## The hero
The toy desk scene. A cursor drops a yellow dot on the mug's logo; the image dims to blue features
("encoded once"); three cutouts lift off in turn (logo, mug, mug and saucer) with score bars; the scene
settles into a static labelled diagram. About six seconds, plays once, Replay button, final frame under
reduced motion.

## Principles
1. One gesture, many papers: every chapter is "what changed about pointing and cutting".
2. The figure is the stage. Two sentences, then the figure, then the idea, then the equation.
3. One prediction prompt per chapter, answered by the figure.
4. Honesty badges on every figure: Toy model / Computed from the paper / Sourced numbers.
5. Write each figure once as a scene; the long page and the atlas mount the same code.

## Rejected defaults
- A photo-real hero or the paper's own figure images (not allowed, and they hide the mechanism).
- Rainbow instance colors for masks: color is reserved for meaning; instances differ by outline pattern
  and label, and the mask is always green.
- Uppercase eyebrows, middle-dot meta strings, arrows on links, gradient washes, shadows on cards.
- Fade-in per section. The hero is the one motion moment; every other animation answers a click.
- three.js: the one 3D figure (SAM 3D) uses the skill's pe3.js canvas renderer.

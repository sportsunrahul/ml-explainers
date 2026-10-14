# Design plan: when a detector gets a text encoder (2026-10-09)

Subject: object detectors that take a sentence as their class list. The world of the subject is the
detection overlay (boxes, labels, scores on a photo) and the prompt field you type into. The reader is an
engineer who builds inspection systems and wants to know when typing a name replaces labelling data.
Primary job: make "where does the text enter the network" visible, chapter after chapter, so the five
designs read as five positions of one tap rather than five unrelated papers.

## Tokens
Color (skill palette, dark first; one meaning each, whole page):
- bg #16181d, fg #e8e6e3, muted #9aa0a6, rule #2a2d34, panel #1d2026, panel2 #23272e
- blue #58c4dd = the image side: pixels, features, region embeddings, predicted boxes
- yellow #f4d35e = the text side: prompts, tokens, text embeddings
- green #83c167 = a match: the region-text score that wins, a positive target, a box a prompt claims
- red #fc6255 = loss and gradient
- brown #c8906d = frozen weights, background, "no object"
Light theme keeps the roles (skill light values).

Type: display = KaTeX_Main (shipped with the math CSS, so headings share the equations' face); body =
Charter 1.125rem/1.7; controls, readouts, captions = system-ui. Prompts are the one special text style:
`span.prompt`, system-ui in yellow inside a thin yellow-outlined pill, because a prompt is literally a
string the model reads, and the reader should see every prompt as the same kind of object in prose, in
figures and in tables. Scale: h1 clamp(2.5rem, 5.6vw, 4.2rem); act titles 2rem; h2 2.1rem; h3 1.3rem.

Layout: page-chrome.js rail (acts + reading progress) from 1180 px, sticky bar below. Prose 40rem,
figures and display equations on the 60rem stage, everything left-aligned to one edge.

```
>= 1180 px                                             < 1180 px
+------------+-----------------------------------+     +-----------------------------+
| I Reads    | 4  ViLD                           |     | 4  ViLD            [menu]   |
| o 1 class  | [tap strip: text enters here  v ] |     +-----------------------------+
| o 2 map    | two sentences of setup            |     | [tap strip]                 |
| II CLIP    | +-------------------------------+ |     | two sentences               |
| * 4 ViLD   | | figure: the stage (60rem)     | |     | +-------------------------+ |
|   ...      | +-------------------------------+ |     | | figure, full width      | |
|            | prose ... equation ... gloss      |     | +-------------------------+ |
+------------+-----------------------------------+     +-----------------------------+
```

## The one bold element: the tap strip
Every model chapter opens with the same thin SVG strip: the image pipeline as a blue lane (image,
backbone, neck, proposals or queries, region features, score) and the text lane in yellow above it.
A yellow drop marks where text enters for this chapter's model; earlier chapters' taps stay as faint
ghosts. Read top to bottom, the strips are the page's argument. The hero ends on the full strip with all
five taps; the family tree at the end colors each paper by its tap position.

## Detection overlay vocabulary (all figures)
- Predicted box: blue corner brackets (four L-shaped corners), not a closed rectangle.
- A box a prompt claims: green brackets plus a small tag "forklift 0.81" (name in yellow, score in fg).
- Background / no object: brown dashed brackets.
- Ground truth, where needed: thin fg rectangle, never bracketed.
Scenes are toy factory and warehouse views drawn from flat shapes (forklift, pallet, crate, worker,
weld seam, scratched panel), because the reader's question is about industrial objects.

## The hero
Seven seconds, canvas, plays once, Replay button, final frame under reduced motion:
1. A warehouse scene; blue brackets appear on four objects (the detector localizes all of them).
2. A classifier block on the right, rows "person, car, chair, ...": three boxes find a row and turn green;
   the forklift's box finds none and turns brown, "background".
3. The rows slide out; a text encoder slides in; the prompt "forklift" types into a pill; a yellow
   vector drops into the block; the forklift box turns green, "forklift".
4. The scene shrinks left and the full tap strip draws: five taps labelled ViLD, OWL-ViT, GLIP,
   Grounding DINO, YOLO-World. Static end frame.

## Principles
1. The figure is the stage; prose is the depth layer. Two sentences, figure, idea, equation, gloss.
2. Same spatial code everywhere: text above, image below, the tap position is depth in the network.
3. One prediction prompt per chapter, answered by the figure.
4. Honesty on the figure: "Toy model" where synthetic, "Computed" where the figure runs the paper's own
   rule or config, "Sourced" where it plots the paper's numbers. Practical-question chapters use sourced
   charts only; any decision rule is labelled "our reading".
5. Depth only once: the embedding sphere (ViLD distillation; reused for the jargon-name failure).

## Rejected defaults
- Uppercase eyebrows, middle-dot meta strings, arrows on links, gradient washes, shadows on cards.
- Monospace for data labels: prompts get their own pill style instead, which carries meaning.
- Fade-in per section; the hero is the one motion moment, every other animation answers a click.
- Closed rectangles for every box: corner brackets make predictions, matches and ground truth distinct.
- Photographic sample images: the papers' images are not ours to embed, and flat drawn scenes keep
  both themes legible.
- three.js: a WebGL context per figure for one point cloud; pe3.js draws it on canvas 2D.

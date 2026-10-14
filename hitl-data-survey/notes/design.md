# Design: Labels in the loop

## Subject, audience, job
Subject: the labeling desk of a vision platform, where a model and a person trade images and labels.
Reader: an engineer who builds such a platform and knows basic deep learning.
Job: make each method feel like a decision the reader can take on Monday ("send these", "re-check those first").

## The one bold thing: the reviewer's marks
Every figure uses the same three hand marks, drawn as slightly uneven pen strokes on top of tiles and points:
- a yellow ring = "sent to a person for a label" (the human's mark),
- a red strike = "this label is wrong",
- a green tick = "kept / believed".
The hero introduces them; every later figure reuses them with the same meaning. Everything else stays quiet.
Images are shown as real CIFAR-10 tiles drawn pixelated (no smoothing), so the dataset reads as data, not decoration.
Toy pools use plain glyphs: class identity is SHAPE (circle, square, triangle), never color, because color is taken.

## Color (3Blue1Brown base tokens, dark first; roles fixed for the whole page)
| token | dark | light | one meaning |
|---|---|---|---|
| --blue | #58c4dd | #1f6f9a | the model: predictions, probabilities, model-chosen picks |
| --yellow | #f4d35e | #8a6700 | the human: labels, label requests, review |
| --green | #83c167 | #3f7d20 | clean / kept / believed |
| --red | #fc6255 | #c0392b | wrong label, loss |
| --brown | #c8906d | #8b5a2b | the unlabeled pool; random picks (drawn from the pool) |
Background, fg, muted, rule, panel: the skill's base values.

## Type
- Headings: KaTeX_Main (embedded with the math CSS; shares the equations' face). Sizes: h1 clamp(2.5rem, 5.6vw, 4.2rem);
  h2 2.1rem; h3 1.3rem Charter 600.
- Body: Charter 1.125rem / 1.7, measure 40rem (page-chrome sets it).
- Figure labels and readouts: system-ui 12-13px, tabular numbers. No monospace labels.
- Sentence case everywhere. No uppercase eyebrows.

## Layout (page-chrome rail on the left at >= 1180 px, sticky bar below)
```
| rail |  h1 "Labels in the loop"                                   |
|      |  hook (2 lines)                     papers as pills         |
|      |  [ hero stage: tiles stream -> model -> labeler -> dataset ] |
|      |                                                             |
|      |  Part one: choosing what to label   (act divider)           |
|      |  h2  chapter                                                |
|      |  two-sentence setup (40rem)                                 |
|      |  [ figure stage 60rem: canvas + controls + readout + badge ]|
|      |  prose (40rem) ... equation (stage width, built by terms)   |
|      |  | The labeling decision: one imperative sentence |  <- yellow rule
```
Left-aligned throughout: prose, figures, equations share one left edge.

## Components
- `.decision` box at the end of every chapter: a yellow left rule, heading "The labeling decision", one or two
  imperative sentences. The platform tie-in lives only here, in generic words.
- `.claim-card` for analysis chapters: Claim / How it was measured / Result / What would have falsified it.
- Badges on every figure (corner): toy, computed (our run or the paper's own algorithm), sourced.
- Pill buttons with hover, pressed (aria-pressed) and focus states.

## Principles
1. One meaning per color, one meaning per mark, for the whole page.
2. Show the decision, not the method: every figure ends in "which images would a person see?"
3. Real images where the claim is about real data; glyphs where the point is geometry.
4. The hero is the only motion that plays by itself; everything else answers a click.

## Rejected defaults
Cream paper with a terracotta accent; an acid-green accent on black; a card grid of "methods"; numbered 01/02/03
markers; gradient washes; fade-in on scroll; colored class labels (would collide with the color roles); monospace
data labels; a stock "pipeline" diagram as the hero.

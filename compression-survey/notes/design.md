# Design: compression survey page

Subject: making a trained vision model cheaper to run on a GPU. The material of the subject is numbers
on a grid (bits, levels, scales), weights that disappear, and kernels that merge. The reader is an
engineer who ships detectors and segmenters; the page's job is intuition first, then a decision.

## Color (one meaning each, page-wide)
| name | dark | light | meaning |
|---|---|---|---|
| ground | #121519 | #f6f7f9 | page background (cool graphite / cool white; not cream) |
| blue | #58c4dd | #1f6f9a | full-precision values, FP32/FP16, the teacher |
| yellow | #f4d35e | #8a6700 | the integer grid, quantized values, the student |
| green | #83c167 | #3f7d20 | what a method keeps or protects: kept weights, salient channels, fused kernels |
| red | #fc6255 | #c0392b | error and loss: rounding, clipping, loss terms, gradients |
| brown | #c8906d | #8b5a2b | removed or frozen: pruned weights, frozen teacher, discarded kernels |
Panels: #191d23 / #eef0f3; rules: #262b33 / #dde1e6.

## Type
- Headings and act dividers: KaTeX_Main (embedded with the math CSS, so it costs nothing and matches
  the equations).
- Body: Charter stack, 1.125rem / 1.7, measure 40rem (page-chrome).
- Figures, controls, readouts, captions: system-ui.
- Integer values in readouts are shown with their bit pattern in a monospace face (they are data:
  `0110 1001`), only inside `.bits` spans. No monospace for labels.
- Scale: h1 clamp(2.5, 5.6vw, 4.2)rem; h2 2.1rem; h3 1.3rem; body 1.125rem; UI .92rem; caption .86rem.

## Layout
```
 rail (>=1180px)   | main (stage 60rem, measure 40rem, left-aligned)
 ----------------- | ---------------------------------------------------
 Act I             | [ hero canvas: 16x16 weight tile -> grid -> prune -> shrink -> fuse ]
  1 Where time...  | Title (KaTeX_Main)       hook sentence
  2 Field          | paper pills (arXiv ids)
 Act II ...        | toc (2 columns)          key + badge note
                   | ---- Act I divider ----
                   | h2  prose (40rem)
                   | [ figure.viz 60rem: canvas + controls + readout + caption ]
                   | prose, equation (stage width), gloss list
                   | .detector aside: "On a detector" one or two sentences -> guide
```
Everything left-aligned to one edge. Figures and display equations break out to the 60rem stage.

## Principles
1. Every figure computes something real. Toy inputs are fine; the arithmetic (FLOPs, bytes, the
   quantizer, KL calibration, GPTQ, FP8 enumeration) is the real algorithm. The badge says which.
2. Bits are the unit. Readouts report bits per value, bytes moved and levels used, not just "error".
3. The guide is the destination. Each chapter ends with an "On a detector" aside that says what the
   chapter means for a deployed detector or segmenter, linking to the guide.
4. One motion moment: the hero. Everything else moves only when clicked.

## Rejected defaults
- Cream paper light theme (#fbfaf7 family): replaced by a cool white that reads as a datasheet.
- Uppercase eyebrow labels, middle-dot meta strings, arrows appended to links, gradient washes, card
  grids, drop shadows on panels, fade-in on scroll.
- A big-number stat hero: the hero is the mechanism, not a statistic.
- Numbered markers on things that are not a sequence (acts and chapters are a real sequence).

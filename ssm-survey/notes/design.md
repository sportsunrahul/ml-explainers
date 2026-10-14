# Design plan: state-space models, a literature survey (2026-10-09)

Subject: how a sequence model remembers. Attention keeps every past token and re-reads all of them;
an SSM keeps one small, fixed-size state and updates it. Every figure on the page is a version of one
picture: tokens arriving along a time axis (blue), a memory that either grows (brown, the KV cache) or
stays the same size (yellow, the state), and what comes out (green). Audience: basic deep learning only,
has the user's own transformers page (attention, KV cache). Primary job: make each mechanism click by
playing with it, and keep the attention comparison in view the whole time.

## Tokens
Color (3Blue1Brown palette from the skill, dark first; one meaning each, whole page):
- bg #16181d, fg #e8e6e3, muted #9aa0a6, rule #2a2d34, panel #1d2026, panel2 #23272e
- blue #58c4dd = input tokens x, the sequence as it arrives
- yellow #f4d35e = the state h, the memory (always drawn as the same small grid of N cells)
- green #83c167 = output y
- red #fc6255 = cost: FLOPs, bytes, error, gradient blow-up
- brown #c8906d = attention and its KV cache, the baseline we compare against
Light theme keeps the roles (the skill's light values). Badges keep the house mapping (toy brown,
computed blue, sourced green); they sit in a corner pill and are words, so they do not compete with the key.

Type: display = KaTeX_Main (embedded with the math); body = Charter 1.125rem/1.7; controls, readouts and
captions = system-ui. Scale: h1 clamp(2.5rem, 5.6vw, 4.2rem); act dividers 2rem; h2 2.1rem; h3 1.3rem;
readouts .92rem. Prose measure 40rem; figures and display equations on the 60rem stage; one left edge.
This matches the user's other explainers on purpose: the pages are a series and read as one shelf.

## The one bold element: the state register
The memory is a fixed grid of yellow cells, drawn the same way in every figure where it appears (the hero,
the recurrence, the 3D trajectory's readout, the scan, the phonebook, the hybrid stack). Its size never
changes; only its colors do. Against it, attention's cache is always a brown block that gets wider as
tokens arrive. The reader learns the page's argument by recognising those two shapes, not by reading.

## Layout (page-chrome.js)
```
>= 1180 px                                            < 1180 px
+-----------+-----------------------------------+     +-----------------------------+
| I Bill    | 6  Mamba: selection               |     | 6  Mamba        [menu]      | sticky
| o 1 bill  | two sentences of setup            |     +-----------------------------+
| o 2 recur | +-------------------------------+ |     | two sentences               |
| II Memory | | figure: the stage (60rem)     | |     | +-------------------------+ |
| o 3 cont. | |  time axis left to right,     | |     | | figure, full width      | |
| III Select| |  register at the right edge   | |     | +-------------------------+ |
| * 6 Mamba | +-------------------------------+ |     | prose ...                   |
|           | prose (40rem) ... equation ...    |     |                             |
+-----------+-----------------------------------+     +-----------------------------+
```
Time always runs left to right; the state register sits at the right end of the time axis (where "now" is).

## The hero (3D, pe3)
Two blocks on one floor. Left, brown: the KV cache of a small Transformer, rows = layers, columns =
tokens; each generated token adds a column of cubes, and a red counter for bytes climbs. Right, yellow:
the SSM state, rows = layers, a fixed width; each token only recolors its cells. Twelve tokens, about
six seconds, then the camera settles to a static three-quarter view with four labels (tokens, KV cache,
state, bytes). Replay button; final frame under reduced motion. Toy sizes, real proportions: the readout
states the bytes for the two configs computed in chapter 1.

## Principles
1. The figure is the stage; prose is the depth layer. Two sentences, figure, idea, equation, gloss.
2. Every chapter states its cost against attention once, in one readout line or one sentence.
3. One prediction prompt per chapter, answered by the figure.
4. Depth only where depth is the idea: the growing cache vs the fixed state (hero), a state that moves
   through 3-D space (continuous time, selection), HiPPO coefficients as a surface over time, and the SSD
   matrix as a surface whose blocks show the chunked algorithm. Everything else is 2D.
5. Honesty on the figure: "Toy model", "Computed" (runs the paper's algorithm or config), "Sourced"
   (plots the paper's numbers). The recall chapter is as visible as the selection chapter.

## Rejected defaults
- Uppercase eyebrows, middle-dot meta strings, arrows on links, gradient washes, shadows on cards.
- Fade-in per section. The hero is the one motion moment; every other animation answers a click.
- An "oscilloscope" look (green-on-black traces, glow): it would steal green from the output role and turn
  instruments into decoration.
- A new font pairing for this page: the user's explainers already form a series; a change would read as
  a different publisher, not a choice.
- three.js: the figures are blocks, lines and surfaces; pe3 on canvas 2D draws them offline with no
  WebGL context and no console noise in QA.

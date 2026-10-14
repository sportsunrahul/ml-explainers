# Design pass: How CLIP works, explained in 3D (2026-10-01)

The page keeps its identity: Bricolage Grotesque + Literata, its color key (--img blue for images,
--txt orange for captions, plus --pos/--neg/--push in the figures), the 3D template and the slow
auto-rotate. The h1 still colors "pictures" and "words" with the key. `hl` is --img, the color of the
page's focus outline.

## Changes
- Hero: the existing pairing animation (image cubes and caption spheres on one sphere) now plays once
  (the page's own 7 s ease) and holds the final clustered layout. Before, `update()` reset `t0` and
  restarted it every 13.3 s. Replay ("Replay training") and the auto-rotate stay; reduced motion shows
  the final frame. The caption gets one sentence that marks it illustrative (the brief's hero rule).
- Section bar: current-section highlight, reading-progress line, and a Sections menu when the nine
  links do not fit (768 and 375; the links fit at 1280 and 1440).
- Equation: the one `.eq` block (two Unicode text lines) is now two KaTeX display lines. I and T carry
  the page's image and caption colors. It points at the similarity-matrix figure that follows it.
- Pressed seg buttons keep readable text when a term lights them (`.seg button[aria-pressed].pe-lit`).
  Without this, the lit "Cosine similarity" button had panel-colored text on the highlight.
- Prose alignment and footer sources: from `common.apply` (one left edge; one list item per reference).

## Equation transcription (original -> TeX), checked one by one
| line | original | TeX |
|---|---|---|
| 1 | s_ij = < I_i, T_j > / tau (I, T unit vectors) | s_{ij} = \langle I_i, T_j \rangle / \tau \qquad (I,\ T\ \text{unit vectors}) |
| 2 | L = 1/2 [ -(1/N) Sum_i log softmax_j(s_ij)_{j=i} - (1/N) Sum_j log softmax_i(s_ij)_{i=j} ] | L = \frac{1}{2}\left[ -\frac{1}{N}\sum_{i} \log \mathrm{softmax}_{j}(s_{ij})_{j=i} \;-\; \frac{1}{N}\sum_{j} \log \mathrm{softmax}_{i}(s_{ij})_{i=j} \right] |

The `<small>` gloss (tau init 0.07, clip at 100, batch 32,768) is unchanged. Colors: only I and T
(t-img, t-txt). The diagonal is green in the figure, but the page has no key that names that color,
so the TeX does not use it.

## Equation links (eq-loss -> f-mat; word -> what lights)
- `s_ij` (cos): the "Cosine similarity" button; every bar glows blue.
- `I_i` (img): the row emoji grow.
- `T_j` (txt): the column caption labels and "captions ->" grow.
- `tau` (tau): the "Temperature" control.
- `L` (loss): the status line with the symmetric InfoNCE loss.
- `softmax_j` (i2t): the "Softmax over captions" button; the front row (one image) glows. The back
  rows hide behind the tall diagonal bars at the default angle.
- `softmax_i` (t2i): the "Softmax over images" button; the first column (one caption) glows.
- `(s_ij)_{j=i}`, `(s_ij)_{i=j}` (diag): the diagonal bars glow in their own green.

## Not changed
- No prose and no number. The only added text is the hero's illustrative sentence and the eq hint.
- Facts row (2021 / 400M / 76.2%): the generic big-number row, but it is content. Kept as is; a
  restyle did not clearly help.
- Hero length stays at the page's 7 s (the brief's "about 6 s" is for new heroes).
- At 375 the second equation line is wider than the box. It scrolls inside `.eq`, as the old text did.
- Pre-existing figure tells, not fixed (they are figure behavior): the f-mat column labels are small
  and crowd at the back; f-force labels can overlap after a random shuffle; the f-zs "dog" label can
  sit on the bird arrow.

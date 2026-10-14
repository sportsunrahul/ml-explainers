# Design plan (2026-10-01)

Subject: diffusion models as ten years of fixes to one idea (destroy structure with noise, learn to undo it).
Audience: a reader with no probability or SDE background. Primary job: make each mechanism click by playing
with it, then read the math.

## Tokens
- Color (3Blue1Brown palette, dark first, one meaning each; the page's existing roles, kept):
  bg #16181d, fg #e8e6e3, yellow #f4d35e (real data), brown #c8906d (pure noise), blue #58c4dd (what the model
  predicts), green #83c167 (generated samples and their paths), red #fc6255 (error, loss). Light theme keeps
  the roles with darker values.
- Type: display = KaTeX_Main (Computer Modern, already embedded for the math: zero bytes, and the headings
  share the face of the equations they introduce); body = Charter; controls, readouts, captions = system-ui.
  Scale: body 1.125rem / 1.7; h3 1.3rem; h2 2.1rem; act title 2rem; h1 clamp(2.5rem, 5.6vw, 4.2rem).
- Layout: page-chrome.js. Fixed left rail (13.5rem) from 1180 px; prose at a 40rem measure; figures and
  display equations on a 60rem stage; everything left-aligned to one edge. Below 1180 px: a sticky bar with
  the current chapter and a chapter menu.

```
>= 1180 px                                         < 1180 px
+-----------+------------------------------+       +---------------------------+
| Diffusion | hero: density band, t ->     |       | 4  Sampling       [menu]  |  sticky bar
|           | data |~~~~~~~~~~~~~~| noise  |       +---------------------------+
| I         | h1, hook, paper pills        |       | hero (taller layout)      |
| o 1 Gen.  | toc in two columns           |       | h1                        |
| II        | ---- Act II divider -----    |       | prose                     |
| * 2 Fwd   | h2, prose 40rem              |       | +-----------------------+ |
| | 3 Noise | +--------------------------+ |       | | figure, full width    | |
| | 4 Samp. | | figure: the stage, 60rem | |       | +-----------------------+ |
+-----------+------------------------------+       +---------------------------+
```

## Principles
- One memorable thing: the hero. Time runs left to right; the data density (three clusters) spreads into
  a Gaussian band; random forward paths go out, green reverse paths come back. It is the picture every
  later chapter refines (Score SDE's view, drawn for a 1D toy), and it is computed, not drawn by hand.
- One motion moment: the hero plays once on load and ends on a static diagram. Every other animation
  answers a click.
- Structure encodes content: numbers only on real sequences (chapters, acts); color only for the five
  meanings.
- Equations point: hover or focus a term and the part of the figure it names lights up. Every equation
  with a figure must pulse something real in that figure.

## Generic defaults rejected (review of the old page)
- Uppercase eyebrow labels (.kicker acts, badges, toc act headers) -> sentence case; acts are titled
  dividers with a muted "Act II" line.
- Middle-dot joined links under the toc -> a short list.
- A paper list as one long sentence of ids -> pill links to the arXiv version read.
- 75ch single column with a rule above every h2 -> rail + measure + stage; no rule above h2.
- Square 6px buttons -> pill controls with hover, pressed and focus states.
- A panel behind the toc and the note -> plain text on the page background.

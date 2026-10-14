# Design notes

Subject: how a network learns by comparing pairs. The characteristic object of the field is the unit
hypersphere with pairs drawn together and everything else spread over it. The page's one bold element is
that sphere: the hero is a circle of embeddings that a real InfoNCE loss rearranges, and the same circle
returns in ch-compare, ch-infonce, ch-negatives and (in 3D) ch-sphere. Everything else stays quiet.

Audience: an engineer who knows basic deep learning and wants to understand, not skim.
Primary job: make "pull the pair, push the rest" and its consequences (temperature, negatives, collapse)
visible and playable.

## Color (dark first; light values in head.html)
- --bg #121519, --panel #191d23: the series shell, kept so the survey pages read as one set.
- --blue #58c4dd: the anchor (view A, the image side).
- --yellow #f4d35e: the positive (view B, the caption).
- --green #83c167: agreement, the pull, alignment.
- --red #fc6255: negatives, their push, the loss and its gradient.
- --brown #c8906d: no gradient: momentum encoder, memory bank, queue, stop-gradient.

## Type
- Headings and act dividers: KaTeX_Main (embedded with the math CSS; the page is math-heavy, so headings
  and equations share one face at no byte cost).
- Body: Charter stack, 1.125rem/1.7, 40rem measure. Controls, captions, readouts: system sans.

## Layout
```
| hero: full-stage canvas, the circle of 24 points animating ............ [Replay] |
| h1 (two lines) / hook / paper pills                                            |
| toc in two columns, acts as dividers                                           |
| color key                                                                      |
| act divider                                                                    |
| h2  | prose (40rem) ...                                                        |
|     | [ figure stage 60rem: canvas, controls, readout, caption ]               |
|     | prose, equation (stage width), gloss, prediction prompt, "In practice"   |
```
Left-aligned throughout; the rail (page-chrome.js) sits on the left above 1180 px.

## Principles
1. One sphere. The same embedding picture recurs, so each chapter changes one thing the reader already knows.
2. Forces as arrows. The pull is always a green spring, the push always red arrows; gradient weights are bar
   lengths, never a legend lookup.
3. Toys are real. Every moving toy runs the actual loss on small data; numbers it prints are computed.
4. Quiet chrome. No eyebrows, no gradients, no shadows; act numbers only because acts are a sequence.

## Rejected defaults
- A new palette per page: the series uses the 3Blue1Brown color roles; changing them would break the
  cross-links to the VLM, OVD and JEPA pages, which use the same meanings for blue and yellow.
- A stats hero (big number + label): the subject is a mechanism, so the hero shows the mechanism.
- Card grids for the method zoo: the family tree carries that, on a time axis.

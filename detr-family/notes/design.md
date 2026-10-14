# Design plan (2026-10-01)

Subject: object detection as set prediction, and the DETR family as a chain of fixes (first make it
train fast, then make it run fast). Audience: an engineer who knows CNNs and transformers. Primary job:
make each mechanism click by playing with it, then read the depth.

## Tokens
- Color (3Blue1Brown palette, one meaning each, dark first): bg #16181d, fg #e8e6e3,
  blue #58c4dd (predictions and queries), yellow #f4d35e (ground truth), green #83c167 (matched, kept,
  positive), red #fc6255 (cost, loss, removed, negative), brown #c8906d ("no object", background,
  baseline models). The light theme keeps the same roles.
- Type: display = KaTeX_Main (embedded with the math CSS, so it costs no bytes and ties headings to the
  equations); body = Charter; controls, readouts and canvas labels = system-ui.
  Scale: body 1.125rem / 1.7; h3 1.3rem; h2 2.1rem; act title 2rem; h1 clamp(2.5rem, 5.6vw, 4.2rem).
- Layout: `page-chrome.js`. A fixed left rail (13.5rem) from 1180 px up; prose holds a 40rem measure;
  figures and display equations break out to the 60rem stage. Below 1180 px: a slim sticky chapter bar
  at the top. On the narrated page the player bar sits at the bottom, under both.

```
>= 1180 px                                          < 1180 px
+-----------+-------------------------------+       +---------------------------+
| The DETR  | hero: slots -> boxes -> match |       | 2 Hungarian match [Chap.] |  sticky chapter bar
| family    | h1, hook, paper chips         |       +---------------------------+
| I DETR    | h2 ...........                |       | h2                        |
| o 1 Set   | prose 40rem ..........        |       | prose                     |
| * 2 Match | +---------------------------+ |       | +-----------------------+ |
| | 3 Query | | figure: the stage, 60rem  | |       | | figure, full width    | |
| | 4 Step  | +---------------------------+ |       | +-----------------------+ |
| II ...    | equation (stage width)        |       |                           |
+-----------+-------------------------------+       +---------------------------+
| narration player (narrated page only)     |       | narration player          |
```

## The hero
DETR's core mechanism in one animation: a column of object-query slots each fires a predicted box at a
toy street scene (car, person, dog). Two of the boxes are duplicates. The ground truth appears, one-to-one
matching links each object to exactly one slot (green), and every other slot, the duplicates included,
is assigned "no object" (brown). It ends on a static diagram: three objects, three matched slots, no NMS.
Plays once on load; Replay button; reduced motion draws the final frame.

## Principles
- The visual is the stage: figures are wider than the prose.
- One motion moment: the hero. Every other animation answers a click, and every click animates
  (400-900 ms, eased); nothing snaps.
- Structure encodes content: numbers only on real sequences (chapters, acts, pipeline stages); color only
  for the five meanings.
- Equations point: hover or focus a term and the part of the figure it names lights up (chapter 2).
- The family tree is a diagram, not a list: papers on a time axis, inheritance as arcs, hover a paper to
  light up what it took from which ancestor.

## Tells removed (review of the old page)
- Uppercase eyebrow labels (.kicker acts, badges) -> sentence case; acts are titled, numbered dividers.
- Middle-dot joined links under the toc, and the boxed toc -> a two-column toc with act headers and a
  short list for the closing sections.
- A run-on line of arXiv ids -> paper chips linked to the versions read.
- "Interlude" as a pseudo-act -> the training step is chapter 4 of Act I, marked as not narrated.
- Figures squeezed into the 75ch text column -> the 60rem stage.

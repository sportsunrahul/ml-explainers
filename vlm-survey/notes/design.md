# Design plan (2026-10-01)

Subject: vision-language models as a chain of fixes. Audience: an engineer who knows transformers and
CNNs. Primary job: make each mechanism click by playing with it, then read the depth.

## Tokens
- Color (the brief: 3Blue1Brown's palette, one meaning each, dark first):
  bg #16181d, fg #e8e6e3, blue #58c4dd (vision), yellow #f4d35e (language), green #83c167 (bridge,
  correct match), red #fc6255 (loss), brown #c8906d (frozen, negatives). Light theme keeps the same roles.
- Type: display = KaTeX_Main (Computer Modern, already embedded for the math, so it costs no bytes and
  ties headings to the equations); body = Charter; controls and readouts = system-ui.
  Scale: body 1.125rem / 1.7; h3 1.3rem; h2 2.1rem; act title 2rem; h1 clamp(2.6rem, 6vw, 4.4rem).
- Layout: a fixed left rail (13.5rem) from 1180 px up; content left-aligned next to it. Prose holds a
  40rem measure; figures break out to a 60rem stage on the right. Below 1180 px: no rail, a slim sticky
  top bar with the current chapter and a chapter menu.

```
>= 1180 px                                         < 1180 px
+-----------+------------------------------+       +---------------------------+
| I         | h2 ............              |       | 3  ViT            [menu]  |  sticky bar
| o 1 CLIP  | prose 40rem ............     |       +---------------------------+
| | 2 SigLIP| prose ........               |       | h2                        |
| * 3 ViT   | +--------------------------+ |       | prose                     |
| | ...     | | figure: the stage, 60rem | |       | +-----------------------+ |
| II        | +--------------------------+ |       | | figure, full width    | |
| o 4 BLIP  | prose ........               |       | +-----------------------+ |
+-----------+------------------------------+       +---------------------------+
```

## Principles
- The visual is the stage: figures are wider than the prose.
- One motion moment: the hero (a photo cuts into patches, a caption into words, and both meet in the
  shared space). Every other animation answers a click.
- Structure encodes content: numbers only on real sequences (chapters, acts, pipeline stages); color only
  for the five meanings.
- Equations point: hover or focus a term and the part of the figure it names lights up.

## Tells removed (review of the old page)
- Uppercase eyebrow labels (.kicker, badges, toc act headers) -> sentence case, acts as titled dividers.
- Middle-dot joined links under the toc -> a short list.
- A long underlined link list as the only navigation -> the rail with progress and current chapter.

# Design pass brief for one "explained in 3D" page

You own exactly one page: a standalone HTML file at the top of `~/paper-explainers/`. Do not edit any other
page, anything under `~/paper-explainers/detr-family/` or `vlm-survey/` or `diffusion-survey/`, anything
under `~/.claude/skills/`, or the files in `~/paper-explainers/shared/` (report a bug in them
instead of fixing it). Put all your work in your slug directory `~/paper-explainers/<slug>/`.

## Reference: the finished pilot
Read these first, in full:
- `~/paper-explainers/transformers/page.py` (the whole pilot transformation, a model for yours)
- `~/paper-explainers/transformers/notes/design.md`
- `~/paper-explainers/shared/common.py`, `chrome.css`, `chrome.js`, `qa.mjs`
- The pilot output: `~/paper-explainers/transformers/Transformers and LLMs, explained in 3D.html` (do not edit it)
All six pages share one template: `nav.top`, `header.hero`, `section` + `.wrap`/`.prose`, `figure.fig`
with a `.canvas` 3D stage built by `stage(id, buildX)`, `.eq` text callouts, and a footer source list.

## Files
1. `cp` the page to `<slug>/original.html` before anything else. Never edit it again.
2. Write `<slug>/page.py`: it reads `original.html`, calls `common.apply(s, hl='var(--<accent>)', extra_css=...)`,
   then makes page-specific replacements with `once()` (each anchor must match exactly once), and writes
   `<slug>/page.src.html`.
3. Build: `cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/prerender-math.mjs "<slug>/page.src.html" "<slug>/<page file name>"`.
   The page keeps its existing file name. Quote every path: the names have spaces and commas.
4. Write `<slug>/notes/design.md` like the pilot's.

## Rules (decided in the pilot; apply them, do not re-decide)
- **Identity stays.** Keep the page's fonts, color tokens, 3D template and auto-rotate. No new palette,
  no new fonts, no layout rebuild.
- **Content stays.** Do not change prose, numbers, figure captions or figure behavior, except as listed here.
  New text you add (hero caption, labels) carries no numbers and says when a picture is illustrative.
- **Section bar** comes from `common.apply` (current section, progress line, menu when the links do not fit).
  Pick `hl` = the page's main accent token.
- **Equations.** Every `.eq` block's formula becomes `<div class="tex tex-display">...</div>` TeX, with the
  original Unicode formula kept in an HTML comment right after it, and the `<small>` gloss left as is.
  Color terms with `\htmlClass{t-<token>}{...}` and add `.t-<token>{color:var(--<token>)}` in extra_css,
  only where the page's own color key already gives that thing a color. Leave inline sub/sup math in prose alone.
  Transcription is the main risk: list every original and its TeX in design.md and check each one.
- **Equations that point.** If an equation sits right before a figure that draws what its terms name,
  give the `.eq` an id and `data-viz="<figure id>"` and wrap terms in `\htmlData{link=<word>, label=<plain words>}{...}`.
  Every link word must light something real: a DOM element in that figure with `data-link="<word>"`
  (a control label, a seg group, the status line), or a 3D layer that reads `s.g('<word>')` in the stage's
  `update()` (scale sprites, raise emissive, show a plane). `common.apply` adds `s.g` to `stage()`.
  Do not invent links where no layer matches. Add the `eq-hint` line ("Hover or tap a term to see it in the
  figure below.") once, under the first pointing equation.
- **Hero.** If the page has no hero figure, add one in `header.hero` like the pilot: a 3D stage `f-hero`
  showing this page's core mechanism, built with the page's existing helpers and tokens. It plays once
  (about 6 s), ends on a static, readable diagram, has `id="hero-replay"`, and shows the final frame when
  `reduced` is true. Give it a narrow layout if the scene is wide (see the pilot's `layout(narrow)`).
  If the page already has a hero that loops (a `t0` reset in `update`), make it play once instead.
  Mark it illustrative in its caption unless it computes something real.
- **Polish and tells.** `common.apply` fixes `.wrap.prose` alignment and splits middle-dot footer lists.
  Note any other generic tell in design.md and fix it only if the fix is small and changes no content.
- ASCII only in code and comments you write (Unicode only inside user-facing string literals that need it).

## QA (you must read the images, not just the JSON)
`cd ~/paper-explainers/.node && node ~/paper-explainers/shared/qa.mjs "<slug>/<page file name>" ~/paper-explainers/<slug>/shots`
It prints a JSON summary and exits 1 on page errors or horizontal overflow. Then Read the PNGs: the hero
mid and final frames (`shots/top/hero-*`), `top-*` and `mid-*` at 1440, 768 and 375 in both themes, the
menu shots, every figure in `shots/figs`, and every pulse in `shots/links` (a contact sheet with PIL is
fastest). Fix what looks off and re-run. Before you report, also re-check that every `.eq` you changed
still says the same thing as its original comment.

## Report back (short)
The QA JSON, the list of changes, the equation table, every link word and what it lights, anything you
chose not to do and why, and anything you are unsure of.

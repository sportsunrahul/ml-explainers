# Design pass: DINO, DINOv2 and DINOv3, explained in 3D (2026-10-01)

The page keeps its identity: Schibsted Grotesk + Source Serif 4, its student/teacher/plum tokens
(teal student, orange teacher, plum loss), the g1/g2/g3 generation colors, the 3D template and its
slow auto-rotate. `hl` is `var(--g1)`, the page's focus-ring color.

## Changes
- Hero: the existing embedding sphere now plays once. Before, it restarted every 15.2 s (the `t0` reset
  in `update`). It still clusters over 8 s, then stays clustered and keeps its slow auto-rotate.
  Replay and the reduced-motion final frame work as before. The scene is round, so it needs no narrow layout.
- Section bar: current-section highlight, reading-progress line, and a Sections menu when the links do
  not fit (here at 768 and 375; at 1280 and 1440 the six links fit). The colored generation dots stay
  in the bar and in the menu.
- Menu dots: the shared `.nav-panel a span{min-width:1.4em}` rule also hit the nav `.dot` spans and
  stretched them into ovals. `.nav-panel a .dot` resets that.
- Equations: four Unicode text blocks rendered with KaTeX, colored by the page's own key (f-pipe and
  f-ibot legends). Two of them point at their figure; every link word lights a real layer.
- Prose alignment: `#choose` and the footer (`.wrap.prose`) now share the left edge of the other sections.
- Readouts: in Schibsted Grotesk, `tabular-nums` also widens `.` `,` `:` to digit width, so readouts showed
  "0 . 053 ," and "(e . g . classification)". `#k-read`, `#g-read` and `#cur-read` (sentence readouts)
  now use normal figures. The slider values (`#c-tau-v`, `#g-it-v`) keep tabular figures, so the next
  control does not shift while you drag.

## Equation transcription (original -> TeX), checked one by one
| id | original | TeX |
|---|---|---|
| (loss) | P_t(x) = softmax((g_theta_t(x) - c) / tau_t)   P_s(x') = softmax(g_theta_s(x') / tau_s) <br> L = Sum_{x in {global}} Sum_{x' != x} - P_t(x) . log P_s(x') | `aligned`, two lines kept; `\frac{g_{\theta_t}(x)-c}{\tau_t}`, `\frac{g_{\theta_s}(x')}{\tau_s}`, `\sum_{x\in\{\text{global}\}}\sum_{x'\neq x} -\,P_t(x)\cdot\log P_s(x')` |
| (EMA) | theta_t <- lambda theta_t + (1 - lambda) theta_s   lambda goes from 0.996 to 1 on a cosine schedule | same, `\leftarrow`; the words in `\text{}`, 0.996 and 1 unchanged |
| eq-koleo | L_KoLeo = -(1/n) Sum_i log d_i,   d_i = min_{j != i} \|x_i - x_j\| | `-\frac{1}{n}\sum_i \log d_i`, `\min_{j\neq i}\lVert x_i - x_j\rVert` |
| eq-gram | L_Gram = \|X_S X_S^T - X_G X_G^T\|_F^2 | `\left\lVert X_S X_S^{\top} - X_G X_G^{\top} \right\rVert_F^{2}` |

Colors: P_t, theta_t and X_G teacher orange; P_s, theta_s and X_S student teal; the DINO loss L plum
("Loss and gradient" in the f-pipe key). L_KoLeo and L_Gram stay ink: no key gives them a color.

## Equation links (word -> what lights)
- eq-koleo -> f-koleo: L_KoLeo (`koleo`) -> the "Apply KoLeo" control and the readout; d_i (`nn`) -> the
  points shrink so the nearest-neighbour lines show, the lines brighten, and the readout lights;
  x_i (`pts`) -> the points grow and glow.
- eq-gram -> f-gram: L_Gram (`gram`) -> the "Gram anchoring from 1M iterations" control;
  X_S X_S^T (`gs`) -> the similarity surface glows and the Gram heatmap gets a ring.
- The hint line sits under eq-koleo, the first pointing equation.

## Not changed
- The DINO loss and EMA equations do not point. f-pipe draws their terms but sits above them, and the
  hint says "below". f-collapse (temperature, centering) is a whole subsection further down.
- X_G X_G^T has no link: no layer draws the Gram teacher's matrix.
- The hero caption: it does not say "illustrative", but the brief's "no prose change" wins, and the
  footer already says the 3D figures are toy data.
- At 375 px the loss, EMA and KoLeo formulas are wider than the box and scroll inside it, as in the
  original and the pilot.
- f-pipe: at 1280 the "cross-entropy loss" label touches the right edge. This is the original camera framing.
No prose, no number, no figure behavior other than the hero loop and the glows.

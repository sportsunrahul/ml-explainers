# Design pass: Post-training and alignment, explained in 3D (2026-10-01)

The page keeps its identity: Onest + Lora, its pol/ref/rew/good/bad tokens (blue policy, grey
reference, orange reward, green preferred, red rejected), the 3D template and its drag-to-rotate stages.
`hl` is `--acc`, the orange that the focus outline already uses.

## Changes
- Bug fix (pre-existing, veto if unwanted): "New comparisons" in the reward-model figure threw
  `Cannot read properties of undefined (reading 'dispose')`, because `ArrowHelper` has no geometry of its
  own. The reset now uses the page's own `disposeTree()`. Probe: one error before, none after.
- Hero (new, illustrative): the page's own France prompt with four candidate replies as horizontal
  probability bars. The policy (blue) starts as the pretrained model. A frozen copy appears as grey
  outlines, green and red reward dots pop in, then four pulls move probability toward the Paris answers.
  Green or red arrows show each pull, and grey springs show the KL penalty stretching back to the
  reference. The policy is `pi_ref * exp(u*K*r)` on made-up numbers, so it never collapses onto one
  answer. Plays once (6 s), ends static, Replay button, final frame under reduced motion. Below canvas
  aspect 1.6, each reply sits above its bar; the hero canvas is 380px tall at 600px and below.
- Section bar: current-section highlight, reading-progress line, and a Sections menu whenever the links
  do not fit (here at every width: ten links).
- Equations: four Unicode text blocks rendered with KaTeX, terms colored by the page tokens. Three point
  at their figure; every link word lights a real layer or control.
- Prose alignment: `.wrap.prose` sections (RLHF, AI feedback, footer) now share the left edge.
- Footer sources: one list item per reference instead of middle-dot joined runs.
- The SFT figure got `id="f-sft"` so QA shoots it.

## Equation transcription (original -> TeX), checked one by one
| id | original | TeX |
|---|---|---|
| eq-bt | P( y_w > y_l \| x ) = sigma( r(x, y_w) - r(x, y_l) ), L_RM = - log sigma( r(x, y_w) - r(x, y_l) ) | same, `\succ`, `\mid`, `\qquad` for the two spaces |
| (rlhf) | max_pi E_{x, y ~ pi} [ r(x, y) ] - beta . KL( pi(.\|x) \|\| pi_ref(.\|x) ) | same, `\max_{\pi}`, `\mathbb{E}`, `\sim`, `\cdot`, `\|` |
| eq-dpo | L_DPO = - log sigma( beta [ log pi(y_w\|x)/pi_ref(y_w\|x) - log pi(y_l\|x)/pi_ref(y_l\|x) ] ) | same, each ratio as `\log\frac{}{}` (log of the ratio) |
| eq-grpo | A_i = ( r_i - mean(r_1..G) ) / std(r_1..G) | `\frac{r_i - \mathrm{mean}(r_{1\ldots G})}{\mathrm{std}(r_{1\ldots G})}` |

Colors: pi = pol, pi_ref = ref, r = rew (the reward-model box), y_w = good, y_l = bad (chosen and
rejected in f-rm and f-dpo). GRPO stays uncolored: its figure shows reward as green or red, so orange
r_i would contradict it, and there is no learned reward model there.

## Equation links (word -> what lights)
- eq-bt -> f-rm: y_w `chosen` -> green spheres grow; y_l `rejected` -> red spheres grow;
  r(x, y_w) - r(x, y_l) `reward` -> the reward surface glows; L_RM `loss` -> the status line.
- eq-dpo -> f-dpo: L_DPO `loss` -> surface glows + status line; beta `beta` -> the beta control;
  chosen ratio `chosen` -> chosen axis lights + its label grows; rejected ratio `rejected` -> same, red.
- eq-grpo -> f-grpo (one paragraph between, as eq-kv in the pilot): A_i `adv` -> advantage sticks widen
  and glow + "sampled group and advantages" grows; r_i `reward` -> sample spheres glow + the
  "% right" labels grow; mean and std `group` -> "Group size G" control + the group label.
  The sticks and spheres exist only after a step; before that, the labels carry the pulse.

## Not changed
- The RLHF objective does not point: f-over draws its terms, but it sits in the next section behind a
  heading and two paragraphs, so the pulse would land off screen.
- Pre-existing, left alone: the GRPO figure crops its bottom labels at 1280 and the last two strategy
  labels overlap; the equations scroll inside their boxes at 375 (as the Unicode originals did).
- No prose, no number, no figure behavior other than the glows and the reset fix.

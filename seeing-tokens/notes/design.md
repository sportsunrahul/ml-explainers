# Design pass: Seeing in tokens: visual tokenizers and VLMs (2026-10-01)

The page keeps its identity: Sora + PT Serif, its enc/code/dec/llm tokens (teal encoder, orange code,
purple decoder, blue language model), the 3D template, and the slow auto-rotate of the codebook stage.
`hl` is `--acc`, the focus-ring colour. On this page `--acc` and `--code` are the same orange.

## Changes
- Hero (new, illustrative): a toy tokenizer. The picture passes a teal encoder into an 8x8 grid of teal
  patch vectors. An orange codebook strip appears. A scan runs row by row: lines join each tile to its
  nearest entry, and the tile takes that entry's colour. A purple decoder then rebuilds the picture.
  The codebook is real k-means (k-means++ seed, as in `buildTok`) on patch colours, ordered dark to light.
  The rebuilt image is the code colours alone, blended by linear texture filtering. The caption says so
  and calls the picture an illustration. Plays once (6.2 s), ends static, Replay button, final frame
  under reduced motion. Below canvas aspect 1.9 the row folds into a "<": encode along the top, decode
  back along the bottom, so the rebuilt image sits under the original.
- Section bar: current section, progress line, and a Sections menu (ten links: menu at every width).
- Equations: two Unicode text blocks rendered with KaTeX. Both point at their figure.
  VQ: `z_e` teal and `z_q`, `e_k`, `e_j` orange, because the f-vq figure draws encoder outputs teal
  and codes orange. The VAE equation is not coloured: its figure colours discs by shape class.
- Prose alignment and footer source lists: from `common.apply`.

## Equation transcription (original -> TeX), checked one by one
| id | original | TeX |
|---|---|---|
| eq-vae | L = E_q(z\|x) [ - log p(x\|z) ] + beta . KL( q(z\|x) \|\| N(0, I) ),  z = mu + sigma (.) eps | `L = \mathbb{E}_{q(z \mid x)}[-\log p(x \mid z)] + \beta \cdot \mathrm{KL}(q(z \mid x) \,\|\, \mathcal{N}(0, I)), \qquad z = \mu + \sigma \odot \varepsilon` |
| eq-vq line 1 | z_q = e_k,  k = argmin_j \|\| z_e - e_j \|\| | `z_q = e_k, \quad k = \operatorname{arg\,min}_{j} \| z_e - e_j \|` (norm not squared, as in the original) |
| eq-vq line 2 | L = \|\| x - D(z_q) \|\|^2 + \|\| sg[z_e] - e_k \|\|^2 + beta \|\| z_e - sg[e_k] \|\|^2 | `L = \| x - D(z_q) \|^2 + \| \mathrm{sg}[z_e] - e_k \|^2 + \beta \| z_e - \mathrm{sg}[e_k] \|^2` (no dot after beta, as in the original) |

The `<br>` between the two VQ lines became two `tex-display` divs. The originals are in HTML comments.

## Equation links (word -> what lights)
- eq-vae -> f-vae: `-log p(x|z)` (recon) -> the decoded thumbnails grow; `beta` -> the "KL weight beta"
  control; `KL` (kl) -> the status line (it reports the average KL to the prior); `q(z|x)` (q) -> the
  Gaussian discs brighten; `N(0, I)` (prior) -> the "Sample from the prior" button and a ring at radius 1
  (the prior's one-sigma contour; prior samples in this figure are drawn with unit variance);
  `sigma` -> the discs widen and the status line (it reports the posterior width sigma).
- eq-vq -> f-vq: `e_k` and `e_j` (code) -> code spheres grow; `k = argmin_j` (assign) -> the
  assignment lines brighten and the status line (codes in use, average quantisation error);
  `z_e` (ze) -> encoder dots grow and glow; the codebook-loss term (cbloss) -> "Train codebook" button
  and the code spheres grow. Codes grow instead of glowing: an emissive glow tints dead grey codes orange.

## Not changed, and why
- No link on `x - D(z_q)` or on the commitment term: f-vq draws no decoder, and nothing in it moves
  the encoder. No link on `mu`, `eps` or `z`: the discs already answer to q and sigma, and prior samples
  exist only after a click. No link to the "Codebook size K" slider: K is not in the equation.
- `D` is not coloured purple. No existing figure draws the decoder in `--dec`; only the new hero does.
- The hero shows no code indices on the tiles: new labels carry no numbers. It has no language-model
  stage: the mechanism here ends at the rebuilt image, and a branch to an LLM would crowd the row.
- No prose, number, caption or figure behaviour changed other than the glows. Text-node diff against
  `original.html`: only the two formulas (now TeX) and the middle-dot footer runs differ.

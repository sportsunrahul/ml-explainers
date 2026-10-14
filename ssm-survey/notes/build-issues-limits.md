# Build issues found while building ch-recall, ch-hybrid, atlas `limits`

- `explainer.head.part` has no `.claim-card` CSS. ch-recall.html carries a scoped `<style>` (before its section) covering `#ch-recall .claim-card, #ch-hybrid .claim-card`; if more chapters use claim cards, move the sam-survey rule into the head.
- Claims marked `uncertain: true` (e.g. `mamba2-mqar-n256-1024-d32` 0.52) fail check-page when wrapped in `data-claim`. The Mamba-2 Fig 8 read-offs (0.95, 0.50, 0.52) are therefore shown as `<span data-approx>about ...</span>` with "read from the plot".
- `zoology-mqar-acc-threshold` etc. fine; `jamba15-act-alpha` is ASCII math (10^-5), shown as TeX, not wrapped.
- The NVIDIA hybrid's "8%" ablation (Fig 4) counts MLP layers in a 24-layer stack; it cannot sit on the viz-hybrid mixer-share axis, so it is prose only.
- Jamba-1.5 Table 1's LLaMA-3.1 405B row (252GB) is 2x the Llama 3 config value; omitted from the page.

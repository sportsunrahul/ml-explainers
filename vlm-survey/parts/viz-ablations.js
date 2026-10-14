<script>
(() => {
  const fig = document.getElementById('viz-ablations'), svg = document.getElementById('ablations-svg');
  const out = document.getElementById('ablations-readout'), note = document.getElementById('ablations-note');
  const c = (text, id) => `<span data-claim="${id}">${text}</span>`;
  // Every plotted value is a page claim (page-ablations-<key>), so each direct label cites its table row.
  const TABS = {
    backbone: { unit: 'Idefics2 average score (higher is better)', max: 80, groups: [
      { name: 'Swap the language model (encoder fixed)', color: '--yellow', bars: [['Llama-1-7B', 'lm-llama', 62.5], ['Mistral-7B', 'lm-mistral', 67.6]] },
      { name: 'Swap the vision encoder (language model fixed)', color: '--blue', bars: [['CLIP-ViT-H at 224 px', 've-clip', 57.4], ['EVA-CLIP-5B at 224 px', 've-eva', 60.2], ['SigLIP-SO400M at 384 px', 've-siglip', 60.7]] }],
      say: 'Swapping the language model moved the average by {0}; swapping the vision encoder, by at most {1}.',
      note: `Not a like-for-like test. The two language models are far apart (${c('35.1% vs 60.1%', 'idefics2-lm-swap-mmlu')} on MMLU), the encoders much closer (${c('78.0% vs 83.2%', 'idefics2-ve-imagenet')} on ImageNet); and SigLIP ran at ${c('384', 'page-ablations-ve-res-siglip')} px against ${c('224', 'page-ablations-ve-res-others')} for the others, which the paper itself says explains part of its TextVQA gain. Cross-attention architecture. Idefics2 v1, Tables 1 and 2.` },
    arch: { unit: 'Idefics2 average score', max: 80, groups: [
      { name: 'Backbones frozen (brown) or trained with LoRA (yellow)', color: '--brown', bars: [
        ['Fully autoregressive, no Perceiver, frozen', 'arch-fa-noperc', 51.8], ['Fully autoregressive, frozen', 'arch-fa-frozen', 60.3],
        ['Cross-attention, frozen', 'arch-ca-frozen', 66.7], ['Cross-attention, LoRA', 'arch-ca-lora', 67.3, '--yellow'], ['Fully autoregressive, LoRA', 'arch-fa-lora', 69.5, '--yellow']] }],
      say: 'Frozen, cross-attention leads by {d:2:1}; once the backbones train, fully autoregressive leads by {d:4:3}.',
      note: `Plotted from the table rows. The paper's prose gives a ${c('7', 'page-ablations-prose-frozen-lead')}-point frozen lead (rows: <span data-computed>6.4</span>) and a ${c('12.9', 'page-ablations-prose-lora-fa')}-point LoRA gain for the fully autoregressive model (rows: <span data-computed>9.2</span>); only its ${c('0.6', 'page-ablations-prose-lora-ca')} for cross-attention matches. Idefics2 v1, Table 3.` },
    connector: { unit: 'Idefics2 average score', max: 80, groups: [
      { name: 'How the patches are pooled (setup not stated)', color: '--green', bars: [['Linear projection', 'conn-linear', 44.5], ['MapNet (MAPL)', 'conn-mapnet', 51.8], ['Perceiver resampler', 'conn-perceiver', 60.3]] },
      { name: 'How deep the Perceiver is', color: '--green', bars: [['1 layer', 'perc-1', 69.3], ['3 layers', 'perc-3', 68.6], ['12 layers', 'perc-12', 69.0]] }],
      say: 'The kind of pooling spans {0}; the Perceiver\'s depth, only {1}.',
      note: 'The kind of pooling matters a lot; the Perceiver\'s depth does not. "Linear projection" is a linear layer over the whole flattened patch sequence that shortens it, not LLaVA\'s per-patch projection. The paper does not state the first group\'s setup (probably frozen backbones: two of its scores equal frozen rows of Table 3); the second group comes from a different setup, so compare within a group only. Idefics2 v1, appendix Tables 11 and 12.' },
    tokens: { unit: 'Idefics2 average score', max: 80, groups: [
      { name: 'Perceiver output size (visual tokens per image)', color: '--blue', bars: [['128 tokens', 'tok-128', 71.2], ['64 tokens', 'tok-64', 71.7]] }],
      say: 'Halving the image from 128 to 64 tokens changed the average by {0}, slightly in favour of 64.',
      note: 'A stronger training setup than the architecture tab, so these absolute scores are not comparable with it. Idefics2 v1, Table 4.' },
    split: { unit: 'benchmark score, same trained model', max: 100, groups: [
      { name: 'TextVQA (reading text)', color: '--blue', bars: [['64 tokens', 'split-textvqa-64', 70.4, null, .45], ['320 tokens (split)', 'split-textvqa-320', 73.0]] },
      { name: 'DocVQA (reading documents)', color: '--blue', bars: [['64 tokens', 'split-docvqa-64', 67.3, null, .45], ['320 tokens (split)', 'split-docvqa-320', 74.0]] },
      { name: 'MMMU (reasoning)', color: '--blue', bars: [['64 tokens', 'split-mmmu-64', 43.5, null, .45], ['320 tokens (split)', 'split-mmmu-320', 43.0]] },
      { name: 'MMBench (general)', color: '--blue', bars: [['64 tokens', 'split-mmb-64', 76.8, null, .45], ['320 tokens (split)', 'split-mmb-320', 76.7]] }],
      say: 'Splitting gains {0} on TextVQA and {1} on DocVQA, and nothing on MMMU or MMBench.',
      note: 'One set of weights, run with one 64-token image or with the image split into four crops plus the whole (<span data-computed>5 &times; 64 = 320</span> tokens). Reading tasks gain; reasoning does not. Idefics2 v1, Table 9 and appendix Table 15.' },
    captions: { unit: 'Molmo-7B-D, pre-trained on each caption source', max: 100, groups: [
      { name: 'cap F1 (caption quality after pre-training)', color: '--yellow', bars: [['ShareGPT4V+o (GPT-4V/4o captions, 158k images)', 'cap-sgpt-f1', 36.3], ['PixMo-Cap: human, spoken', 'cap-human-f1', 54.1], ['PixMo images, captioned by GPT-4o', 'cap-gpt4o-f1', 52.9]] },
      { name: '11-benchmark average after fine-tuning', color: '--yellow', bars: [['ShareGPT4V+o', 'cap-sgpt-avg', 74.9], ['PixMo-Cap: human, spoken', 'cap-human-avg', 76.9], ['PixMo images, captioned by GPT-4o', 'cap-gpt4o-avg', 77.5]] },
      { name: 'Human-preference Elo (no meaningful zero, so dots)', kind: 'dot', lo: 960, hi: 1040, color: '--yellow', bars: [['no PixMo-Cap at all', 'elo-nocap', 990], ['PixMo-Cap: human, spoken', 'elo-default', 1014], ['PixMo images, captioned by GPT-4o', 'elo-gpt4o', 1018]] }],
      say: 'Human spoken captions and GPT-4o captions of the same images are within {0:1} cap F1, {1:1} on the benchmark average and {2:1} Elo of each other: a tie.',
      note: `cap F1 is Molmo's own metric: GPT-4o judges each generated caption against human spoken transcripts collected the same way as PixMo-Cap, which may favour human-style captions; the benchmark average favours GPT-4o. ShareGPT4V+o covers only ${c('158k', 'page-ablations-sgpt-images')} images; the paper compares it with the ${c('178k', 'page-ablations-scale-178k-images')}-image row of the next tab (a different setup, since that row also trims PixMo-Cap from fine-tuning) and finds it still behind. The Elo axis runs from <span data-computed>960 to 1040</span>. Molmo v2, Tables 3b and 5.` },
    scale: { unit: 'Molmo-7B-D, 11-benchmark average', max: 100, groups: [
      { name: 'PixMo-Cap images (pre-training and fine-tuning)', color: '--yellow', bars: [['none', 'scale-0', 74.9], ['89k', 'scale-89k', 75.5], ['178k', 'scale-178k', 76.3], ['356k', 'scale-356k', 76.2], ['712k', 'scale-712k', 76.9]] },
      { name: 'How the image is cut into crops', color: '--blue', bars: [['one crop', 'crop-single', 62.8], ['several crops, no overlap', 'crop-nooverlap', 75.7], ['several crops, overlapping', 'crop-overlap', 76.9]] }],
      say: 'All of PixMo-Cap adds {0} over none; multi-crop adds {1} over a single crop.',
      note: 'More human captions keep helping, slowly. Cropping is the bigger lever: a single low-resolution view loses <span data-computed>14.1</span> points, and letting crops overlap (so border patches see their neighbours) adds a little more. Molmo v2, Tables 3a and 2d.' },
  };
  const ORDER = Object.keys(TABS);
  const S = { tab: 'backbone', u: 1 };
  let stop = () => {};
  const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;');

  function draw() {
    const T = TABS[S.tab], W = svg.clientWidth || 600, small = W < 520, fs = small ? 11.5 : 12.5;
    const F = PE.css('--fg'), M = PE.css('--muted'), R = PE.css('--rule');
    const x0 = 6, valW = 52, bw = W - x0 - valW - 6, barH = 13;
    let y = 22, h = '';
    h += `<text x="${x0}" y="14" fill="${M}" font-size="${fs - 1}" font-family="system-ui, sans-serif">${esc(T.unit)}</text>`;
    T.groups.forEach(g => {
      y += 18; h += `<text x="${x0}" y="${y}" fill="${F}" font-size="${fs + .5}" font-weight="700" font-family="system-ui, sans-serif">${esc(g.name)}</text>`; y += 6;
      g.bars.forEach(([name, key, v, col, alpha]) => {
        const colour = PE.css(col || g.color);
        y += fs + 4; h += `<text x="${x0}" y="${y}" fill="${F}" font-size="${fs}" font-family="system-ui, sans-serif">${esc(name)}</text>`; y += 5;
        if (g.kind === 'dot') {
          const px = x0 + bw * (PE.lerp(g.lo, v, S.u) - g.lo) / (g.hi - g.lo);
          h += `<line x1="${x0}" x2="${x0 + bw}" y1="${y + barH / 2}" y2="${y + barH / 2}" stroke="${R}" stroke-width="2"/>`;
          h += `<circle cx="${px}" cy="${y + barH / 2}" r="6" fill="${colour}" stroke="${PE.css('--bg')}" stroke-width="2"><title>${esc(name)}: ${v}</title></circle>`;
          h += `<text data-claim="page-ablations-${key}" x="${px + 11}" y="${y + barH - 2}" fill="${F}" font-size="${fs}" font-family="system-ui, sans-serif">${v}</text>`;
        } else {
          const w = Math.max(2, bw * v / T.max * S.u);
          h += `<rect x="${x0}" y="${y}" width="${w}" height="${barH}" rx="3" fill="${colour}" fill-opacity="${alpha || 1}"><title>${esc(name)}: ${v.toFixed(1)}</title></rect>`;
          h += `<text data-claim="page-ablations-${key}" x="${x0 + w + 6}" y="${y + barH - 2}" fill="${F}" font-size="${fs}" font-family="system-ui, sans-serif">${v.toFixed(1)}</text>`;
        }
        y += barH + 4;
      });
    });
    y += 8; svg.setAttribute('viewBox', `0 0 ${W} ${y}`); svg.style.height = y + 'px'; svg.innerHTML = h;
  }
  function say() {
    const T = TABS[S.tab], spread = g => { const v = g.bars.map(b => b[2]); return Math.max(...v) - Math.min(...v); };
    const flat = T.groups.flatMap(g => g.bars.map(b => b[2]));
    out.textContent = T.say
      .replace(/\{d:(\d):(\d)\}/g, (_, a, b) => (flat[+a] - flat[+b]).toFixed(1) + ' points')
      .replace(/\{(\d)(:1)?\}/g, (_, i, one) => { const g = T.groups[+i]; if (!one) return spread(g).toFixed(1) + ' points'; return Math.abs(g.bars[1][2] - g.bars[2][2]).toFixed(1); });
  }
  function show(tab, still) {
    stop(); S.tab = tab;
    ORDER.forEach(k => document.getElementById('btn-ablations-' + k).setAttribute('aria-pressed', String(k === tab)));
    note.innerHTML = TABS[tab].note; say();
    if (still) { S.u = 1; draw(); } else stop = PE.animate(700, u => { S.u = u; draw(); });
  }
  ORDER.forEach(k => document.getElementById('btn-ablations-' + k).addEventListener('click', () => show(k)));
  document.getElementById('btn-ablations-reset').addEventListener('click', () => show('backbone', true));
  new ResizeObserver(() => draw()).observe(svg);
  PE.onTheme(draw);
  show('backbone', true);
})();
</script>

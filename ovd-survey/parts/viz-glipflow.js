<script>
(() => {
  const x = '×';
  // Depths, widths and sizes from the official configs (code-only); location count is our reading at the test size.
  const V = {
    glipt: { name: 'GLIP-T', back: 'Swin-T', L: 6, gpus: 32, data: 'Objects365, GoldG, Cap4M' },
    glipl: { name: 'GLIP-L', back: 'Swin-L', L: 8, gpus: 64, data: 'FourODs, GoldG, Cap24M' },
  };
  const N = '22,400';
  const SH = {
    img: () => `3${x}800${x}1333 (test)`, prompt: () => 'M ≤ 256 tokens', bert: () => `P⁰: M${x}d_text`,
    swin: () => 'strides 8, 16, 32', fpn: () => `5 levels ${x} 256 ch`, xmha: () => `Attn: N${x}M per head`,
    bertl: () => `P: M${x}d_text`, dyhead: () => `O: ~${N}${x}256`, score: () => `~${N}${x}M`, box: () => `~${N}${x}4`, loss: () => 'one number' };
  const PARAMS = { swin: 'backbone|lr 1e-4', bert: 'BERT weights|lr 1e-5', fpn: 'FPN convs', xmha: 'W(q), W(v), W(out)|both sides',
    bertl: 'added layers', dyhead: 'DyHead modules', box: 'regression convs' };
  ArchFlow.mount({
    prefix: 'glipflow', grid: { cols: 7, rows: 3 },
    boxes: {
      prompt: { c: 0, r: 0, lane: 'yellow', title: ['prompt'], glyph: 'text', data: true },
      bert: { c: 1, r: 0, lane: 'yellow', title: ['BERT-base'], glyph: 'layers' },
      img: { c: 0, r: 2, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      swin: { c: 1, r: 2, lane: 'blue', title: ['backbone'], glyph: 'grid', cls: false },
      fpn: { c: 2, r: 2, lane: 'blue', title: ['FPN'], glyph: 'layers' },
      xmha: { c: 3, r: 1, lane: 'mixed', title: ['X-MHA'], glyph: 'seq', pattern: 'yyybbb' },
      bertl: { c: 4, r: 0, lane: 'yellow', title: ['added BERT', 'layers'], glyph: 'layers' },
      dyhead: { c: 4, r: 2, lane: 'blue', title: ['DyHead', 'modules'], glyph: 'layers' },
      score: { c: 5, r: 1, lane: 'green', title: ['alignment', 'O Pᵀ'], glyph: 'matrix' },
      box: { c: 5, r: 2, lane: 'blue', title: ['box head'], glyph: 'mlp' },
      loss: { c: 6, r: 1, lane: 'red', title: ['focal loss', '+ box loss'], glyph: 'loss' },
    },
    edges: [['prompt', 'bert'], ['img', 'swin'], ['swin', 'fpn'], ['bert', 'xmha'], ['fpn', 'xmha'], ['bert', 'bertl'], ['fpn', 'dyhead'],
      ['xmha', 'bertl'], ['xmha', 'dyhead'], ['bertl', 'score'], ['dyhead', 'score'], ['dyhead', 'box'], ['score', 'loss'], ['box', 'loss']],
    stages: [
      { key: 'inputs', show: ['img', 'prompt'] }, { key: 'encoders', show: ['swin', 'fpn', 'bert'] }, { key: 'fusion', show: ['xmha', 'dyhead', 'bertl'] },
      { key: 'heads', show: ['score', 'box'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'glipt',
    shape: id => SH[id](),
    sub: (id, v) => id === 'swin' ? `${v.back}|strides 8, 16, 32` : id === 'dyhead' ? `L = ${v.L}|O: ~${N}${x}256` : id === 'bertl' ? 'L layers (count|not stated)' : null,
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, v) => ({
      overview: `${v.name}: ${v.back} and BERT-base read the image and the prompt; X-MHA joins them before each of the ${v.L} DyHead modules (official code); the score is every location against every token. Pre-trained on ${v.data}.`,
      inputs: `Step 1: one image, short side 480 to 800 in training (multi-scale), 800 ${x} 1333 at test (official code). One prompt of at most 256 tokens: up to 85 category names, or a caption.`,
      encoders: `Step 2: ${v.back} gives feature maps at strides 8, 16, 32; the FPN makes 5 levels (strides 8 to 128) of 256 channels. BERT gives P⁰, one vector per token: M ${x} d_text (width kept symbolic).`,
      fusion: `Step 3: ${v.L} rounds of X-MHA, a DyHead module and an added BERT layer. Attn is N ${x} M per head, with N about ${N} locations at the test size (our reading).`,
      heads: `Step 4: S_ground = O Pᵀ is about ${N} ${x} M; the box head regresses 4 numbers per location.`,
      loss: 'Step 5: focal sigmoid loss of S_ground against T\' over every cell, plus the localization loss on the positive locations.',
      backward: 'Backward: red outlines on every box with weights. The gradient reaches BERT twice, through the score and through every X-MHA. Nothing is frozen.',
      update: `Update: AdamW (official code), lr 1e-4 and 1e-5 for BERT, batch 64 on ${v.gpus} GPUs; both rates drop ${x}0.1 at 67% and 89% of training.`,
    }[s]),
  });
})();
</script>

(() => {
  const x = 'x';
  // ViT sizes: ViT paper Table 1. Recipe per size: SAM v1 App. A. Params: SAM Fig. 13 axis labels.
  const V = {
    b: { name: 'ViT-B', D: 768, L: 12, H: 12, par: '91M', lr: '8e-4', ld: 0.6, dp: 0.6, it: '180k', bs: 128, gpu: 128 },
    l: { name: 'ViT-L', D: 1024, L: 24, H: 16, par: '308M', lr: '4e-4', ld: 0.8, dp: 0.4, it: '180k', bs: 128, gpu: 128 },
    h: { name: 'ViT-H', D: 1280, L: 32, H: 16, par: '636M', lr: '8e-4', ld: 0.8, dp: 0.4, it: '90k', bs: 256, gpu: 256 },
  };
  const R = 1024, P = 16, G = R / P, N = G * G;
  const SH = {
    img: () => `3${x}${R}${x}${R}`, enc: m => `${N}${x}${m.D} -> neck`, emb: () => `256${x}${G}${x}${G}`,
    pts: () => 'click: 1 token|box: 2 tokens', penc: () => 'N_p tokens of 256', mprompt: () => `1${x}256${x}256 logits`, mconv: () => `256${x}${G}${x}${G} (added)`,
    dec: () => `(N_p+5)${x}256 tokens|${N}${x}256 image`, up: () => `32${x}256${x}256`, mlp: () => `3 vectors of 32`, masks: () => `3${x}256${x}256 + 3 scores`, loss: () => 'one number' };
  const PARAMS = { enc: 'all ViT layers, neck convs', penc: 'point and corner embeddings, no-mask embedding', mconv: 'mask-prompt convs', dec: 'attention and MLP layers, output tokens', up: 'transposed convs', mlp: 'token MLPs, score head' };
  ArchFlow.mount({
    prefix: 'samflow', grid: { cols: 7, rows: 3 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      enc: { c: 1, r: 0, lane: 'blue', title: ['image', 'encoder'], glyph: 'layers' },
      emb: { c: 2, r: 0, lane: 'blue', title: ['image', 'embedding'], glyph: 'grid', cls: false },
      mprompt: { c: 0, r: 1, lane: 'yellow', title: ['previous', 'mask logits'], glyph: 'grid', data: true, cls: false },
      mconv: { c: 1, r: 1, lane: 'yellow', title: ['mask', 'convs'], glyph: 'layers' },
      pts: { c: 0, r: 2, lane: 'yellow', title: ['clicks,', 'box'], glyph: 'tokens', data: true },
      penc: { c: 1, r: 2, lane: 'yellow', title: ['prompt', 'encoder'], glyph: 'tokens' },
      dec: { c: 3, r: 1, lane: 'green', title: ['two-way', 'transformer'], glyph: 'layers' },
      up: { c: 4, r: 0, lane: 'green', title: ['upscale', '4x'], glyph: 'layers' },
      mlp: { c: 4, r: 2, lane: 'green', title: ['token MLPs,', 'score head'], glyph: 'vector' },
      masks: { c: 5, r: 1, lane: 'green', title: ['3 masks', '+ scores'], glyph: 'grid', cls: false },
      loss: { c: 6, r: 1, lane: 'red', title: ['focal + dice', '+ score loss'], glyph: 'loss' },
    },
    edges: [['img', 'enc'], ['enc', 'emb'], ['emb', 'dec'], ['mprompt', 'mconv'], ['mconv', 'dec'], ['pts', 'penc'], ['penc', 'dec'], ['dec', 'up'], ['dec', 'mlp'], ['up', 'masks'], ['mlp', 'masks'], ['masks', 'loss']],
    stages: [
      { key: 'image', show: ['img'] }, { key: 'encoder', show: ['enc', 'emb'] }, { key: 'prompts', show: ['pts', 'penc', 'mprompt', 'mconv'] },
      { key: 'decoder', show: ['dec'] }, { key: 'masks', show: ['up', 'mlp', 'masks'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'h',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'enc' ? `${m.L} layers, ${m.H} heads, width ${m.D}` : id === 'dec' ? '2 layers, 8 heads, width 256' : null,
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => ({
      overview: `${m.name} (${m.par} parameters in SAM's Fig. 13): a ${R} ${x} ${R} image becomes ${G} ${x} ${G} = ${N} tokens of width ${m.D}, then a 256 ${x} ${G} ${x} ${G} embedding. Everything right of the embedding runs per prompt.`,
      image: `Step 1: one image, rescaled and padded to 3 ${x} ${R} ${x} ${R}. Batch ${m.bs} images for ${m.name}.`,
      encoder: `Step 2: ${R}/${P} = ${G} patches per side, ${N} tokens of width ${m.D}; ${m.L} layers with ${m.H} heads; the neck (1${x}1 then 3${x}3 conv) maps ${m.D} -> 256 channels: 256 ${x} ${G} ${x} ${G}.`,
      prompts: `Step 3: N_p prompt tokens of 256 (1 per click, 2 per box). Previous logits 256 ${x} 256 -> two stride-2 convs (4, 16 channels) -> 1${x}1 conv -> 256 ${x} ${G} ${x} ${G}, added to the image embedding.`,
      decoder: `Step 4: N_p + 5 tokens and ${N} image vectors, width 256. Each of 2 layers: token self-attention, tokens -> image, token MLP (2048), image -> tokens. Cross-attention runs at 128 channels.`,
      masks: `Step 5: upsample 256 ${x} ${G} ${x} ${G} -> 64 ch at ${G * 2} -> 32 ch at ${G * 4}: 32 ${x} 256 ${x} 256. Each mask token -> 3-layer MLP -> 32 numbers; dot product per location -> 3 masks of 256 ${x} 256, plus 3 predicted IoUs.`,
      loss: 'Step 6: 20 focal + 1 dice per mask against the true mask; keep the lowest of the three (single prompt); add the score loss with weight 1.0. One number.',
      backward: `Backward: every box turns red. The gradient reaches the image encoder too, all ${m.L} layers: nothing in SAM is frozen.`,
      update: `Update: AdamW, warmup 250 iterations, learning rate ${m.lr}, layer-wise decay ${m.ld}, drop path ${m.dp}, weight decay 0.1, ${m.it} iterations at batch ${m.bs} on ${m.gpu} GPUs.`,
    }[s]),
  });
})();

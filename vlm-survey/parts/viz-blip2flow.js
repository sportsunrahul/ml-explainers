<script>
(() => {
  // Two mounts: ArchFlow's box set and edges are fixed per mount, and BLIP-2's two stages have different graphs.
  const x = '×', R = 224, P = 14, NQ = 32, DQ = 768, n = (R / P) ** 2 + 1;
  const V1 = {
    l: { name: 'ViT-L/14 (CLIP)', D: '1024', batch: 2320 },
    g: { name: 'ViT-g/14 (EVA-CLIP)', D: 'd_g', batch: 1680 },
  };
  const SH1 = {
    img: () => `N${x}3${x}${R}${x}${R}`, vit: m => `N${x}${n}${x}${m.D}`, qry: () => `${NQ}${x}${DQ}`, txt: () => 'N strings', tok: () => `N${x}L${x}${DQ}`,
    qf: () => `Z: N${x}${NQ}${x}${DQ}|text: N${x}L${x}${DQ}`, sim: () => `N${x}N`, itm: () => `N'${x}2`, itg: () => `N${x}L${x}vocab`,
    litc: () => 'one number', litm: () => 'one number', litg: () => 'one number', loss: () => 'sum of three',
  };
  const SUB1 = { qry: () => `${NQ}${x}${DQ}, same|for every image`, sim: () => `N${x}N: max of|${NQ} per pair`, itm: () => `N'${x}2: mean|over ${NQ} queries` };
  const PAR1 = { qry: 'the queries|themselves', tok: 'word emb', qf: 'SA, CA, FFN', itm: '2-class linear', itg: 'output layer' };
  ArchFlow.mount({
    prefix: 'blip2flow', grid: { cols: 6, rows: 3 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['photos'], glyph: 'photos', data: true },
      vit: { c: 1, r: 0, lane: 'blue', title: ['image', 'encoder'], glyph: 'layers' },
      txt: { c: 0, r: 2, lane: 'yellow', title: ['captions'], glyph: 'text', data: true },
      tok: { c: 1, r: 2, lane: 'yellow', title: ['tokens', '+ embed'], glyph: 'tokens' },
      qf: { c: 2, r: 1, lane: 'green', title: ['Q-Former', '(3 masks)'], glyph: 'layers' },
      qry: { c: 2, r: 2, lane: 'green', title: ['learned', 'queries'], glyph: 'latents' },
      sim: { c: 3, r: 0, lane: 'green', title: ['ITC', 'similarity'], glyph: 'matrix' },
      itm: { c: 3, r: 1, lane: 'green', title: ['ITM', 'classifier'], glyph: 'vector' },
      itg: { c: 3, r: 2, lane: 'yellow', title: ['ITG', 'next word'], glyph: 'tokens' },
      litc: { c: 4, r: 0, lane: 'red', title: ['ITC loss'], glyph: 'loss' },
      litm: { c: 4, r: 1, lane: 'red', title: ['ITM loss'], glyph: 'loss' },
      litg: { c: 4, r: 2, lane: 'red', title: ['ITG loss'], glyph: 'loss' },
      loss: { c: 5, r: 1, lane: 'red', title: ['stage 1', 'loss'], glyph: 'loss' },
    },
    edges: [['img', 'vit'], ['vit', 'qf'], ['qry', 'qf'], ['txt', 'tok'], ['tok', 'qf'], ['qf', 'sim'], ['qf', 'itm'], ['qf', 'itg'],
      ['sim', 'litc'], ['itm', 'litm'], ['itg', 'litg'], ['litc', 'loss'], ['litm', 'loss'], ['litg', 'loss']],
    stages: [
      { key: 'batch', show: ['img', 'txt'] }, { key: 'vit', show: ['vit'] }, { key: 'inputs', show: ['qry', 'tok'] }, { key: 'qformer', show: ['qf'] },
      { key: 'heads', show: ['sim', 'itm', 'itg'] }, { key: 'loss', show: ['litc', 'litm', 'litg', 'loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V1, variant: 'g',
    shape: (id, m) => SH1[id](m),
    sub: (id, m) => SUB1[id] ? SUB1[id](m) : null,
    params: id => PAR1[id] || null,
    frozen: id => id === 'vit',
    readout: (s, m) => ({
      overview: `Stage 1 with ${m.name}: the frozen encoder turns a ${R} ${x} ${R} image into ${n} ${x} ${m.D} features; the Q-Former turns them into ${NQ} ${x} ${DQ}, whatever the image.`,
      batch: `Step 1: N image-caption pairs, images ${SH1.img()}. Stage-1 batch for ${m.name}: N = ${m.batch}.`,
      vit: `Step 2: frozen ${m.name}, last layer removed: ${(R / P)} ${x} ${(R / P)} = ${(R / P) ** 2} patches + [CLS] = ${n} rows of width ${m.D}, ${SH1.vit(m)}.`,
      inputs: `Step 3: the ${NQ} learned queries, ${NQ} ${x} ${DQ}, copied for every image in the batch (N ${x} ${NQ} ${x} ${DQ}); the caption as BERT tokens, ${SH1.tok()}, headed by [CLS] (ITC, ITM) or [DEC] (ITG).`,
      qformer: `Step 4: queries and text in one sequence of ${NQ} + L through shared self-attention; only the queries cross-attend to the ${n} ${x} ${m.D} image features. Three passes, one per mask. Out: Z, N ${x} ${NQ} ${x} ${DQ}, and text states, ${SH1.tok()}.`,
      heads: `Step 5: ITC compares each of the ${NQ} rows of Z with the text [CLS] and keeps the best (N ${x} N); ITM puts each row through a 2-class linear layer and averages (N' ${x} 2); ITG predicts each next word (N ${x} L ${x} vocab).`,
      loss: 'Step 6: contrastive cross-entropy with in-batch negatives, binary cross-entropy on matched vs hard-negative pairs, next-word cross-entropy on the caption; jointly optimized.',
      backward: 'Backward: the gradient reaches the heads, every Q-Former layer, the word embeddings and the queries themselves. It stops at the frozen image encoder: nothing upstream of it trains.',
      update: `Update: AdamW on the Q-Former, its heads, the word embeddings and the ${NQ} queries only; ${m.name} never changes. ${m.name} batch ${m.batch}, 250k steps.`,
    }[s]),
  });

  const V2 = {
    opt: { name: 'OPT (decoder-only)', batch: 1920, prec: 'FP16' },
    t5: { name: 'FlanT5 (encoder-decoder)', batch: 1520, prec: 'BFloat16' },
  };
  const t5 = m => m === V2.t5;
  const SH2 = {
    img: () => `N${x}3${x}${R}${x}${R}`, vit: () => `N${x}${n}${x}d_v`, qry: () => `${NQ}${x}${DQ}`, qf: () => `Z: N${x}${NQ}${x}${DQ}`,
    fc: () => `N${x}${NQ}${x}d_LLM`, txt: () => 'N strings', tok: m => t5(m) ? `prefix + suffix|${x} d_LLM` : `N${x}L${x}d_LLM`,
    llm: m => t5(m) ? `enc: ${NQ} + prefix|dec: suffix` : `N${x}(${NQ}+L)${x}d_LLM`, loss: m => t5(m) ? 'suffix tokens' : 'text tokens',
  };
  const PAR2 = { qry: 'the queries', qf: 'SA, CA, FFN', fc: 'W_FC, b' };
  ArchFlow.mount({
    prefix: 'blip2flow-s2', grid: { cols: 7, rows: 2 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['photos'], glyph: 'photos', data: true },
      vit: { c: 1, r: 0, lane: 'blue', title: ['image', 'encoder'], glyph: 'layers' },
      qf: { c: 2, r: 0, lane: 'green', title: ['Q-Former', '(stage 1 init)'], glyph: 'layers' },
      qry: { c: 2, r: 1, lane: 'green', title: ['learned', 'queries'], glyph: 'latents' },
      fc: { c: 3, r: 0, lane: 'green', title: ['FC', 'layer'], glyph: 'mlp' },
      txt: { c: 3, r: 1, lane: 'yellow', title: ['captions'], glyph: 'text', data: true },
      tok: { c: 4, r: 1, lane: 'yellow', title: ['LLM token', 'embeddings'], glyph: 'tokens' },
      llm: { c: 5, r: .5, lane: 'yellow', title: ['LLM'], glyph: 'seq', pattern: 'ggggyyyyy' },
      loss: { c: 6, r: .5, lane: 'red', title: ['LM loss'], glyph: 'loss' },
    },
    edges: [['img', 'vit'], ['vit', 'qf'], ['qry', 'qf'], ['qf', 'fc'], ['fc', 'llm'], ['txt', 'tok'], ['tok', 'llm'], ['llm', 'loss']],
    stages: [
      { key: 'batch', show: ['img', 'txt'] }, { key: 'extract', show: ['vit', 'qry', 'qf'] }, { key: 'project', show: ['fc'] },
      { key: 'splice', show: ['tok'] }, { key: 'llm', show: ['llm'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V2, variant: 'opt',
    shape: (id, m) => SH2[id](m),
    sub: (id, m) => id === 'qry' ? `${NQ}${x}${DQ}` : null,
    params: id => PAR2[id] || null,
    frozen: id => id === 'vit' || id === 'tok' || id === 'llm',
    readout: (s, m) => ({
      overview: `Stage 2 with ${m.name}: image to ${NQ} query vectors (as in stage 1), one FC layer to the LLM's width, then the frozen LLM reads them as a prefix. d_v is the frozen ViT's width, d_LLM the LLM's; BLIP-2 prints neither for ViT-g or the LLMs.`,
      batch: `Step 1: N image-caption pairs, images ${SH2.img()}. Stage-2 batch for ${m.name}: N = ${m.batch}.`,
      extract: `Step 2: frozen ViT (${SH2.vit()}, d_v = 1024 for ViT-L/14) and the Q-Former from stage 1, queries only, no text: Z is N ${x} ${NQ} ${x} ${DQ}.`,
      project: `Step 3: one fully connected layer, ${DQ} to d_LLM: N ${x} ${NQ} ${x} d_LLM. These ${NQ} vectors are the soft visual prompt.`,
      splice: t5(m) ? 'Step 4: the caption is split in two. The prefix goes to the encoder after the 32 prompt vectors; the suffix is the decoder\'s target. Both are embedded by FlanT5\'s own frozen table.'
        : `Step 4: the caption embedded by OPT's own frozen table, N ${x} L ${x} d_LLM, placed after the ${NQ} prompt vectors: N ${x} (${NQ} + L) ${x} d_LLM.`,
      llm: t5(m) ? `Step 5: FlanT5 (in ${m.prec}) reads [prompt; prefix] in its encoder, and its decoder generates the suffix.`
        : `Step 5: OPT (in ${m.prec}) runs its causal stack over [prompt; caption]; every caption position predicts the next token.`,
      loss: t5(m) ? 'Step 6: prefix language modeling: cross-entropy on the suffix tokens only.' : 'Step 6: language modeling: cross-entropy on the caption tokens given the visual prompt.',
      backward: 'Backward: the gradient must pass through the whole frozen LLM to reach the FC layer, then the Q-Former and its queries. It stops at the frozen ViT and at the LLM\'s token table: nothing trainable lies behind them.',
      update: `Update: AdamW on the FC layer, the Q-Former and the queries; the ViT and ${m.name.split(' ')[0]} never change. Batch ${m.batch}, 80k steps.`,
    }[s]),
  });
})();
</script>

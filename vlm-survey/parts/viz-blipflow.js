<script>
(() => {
  const x = '×', R = 224;
  // ViT-B/16 and ViT-L/16 (BLIP Sec. 4.1), widths and depths from ViT's Table 1; text width 768 is BERT-base's hidden size.
  const V = {
    b: { name: 'ViT-B/16', P: 16, D: 768, layers: 12, batch: 2880, lr: '3e-4' },
    l: { name: 'ViT-L/16', P: 16, D: 1024, layers: 24, batch: 2400, lr: '2e-4' },
  };
  const T = 768, n = m => (R / m.P) ** 2 + 1;
  const SH = {
    img: () => `N${x}3${x}${R}${x}${R}`, pat: m => `N${x}${n(m)}${x}${m.D}`, vit: m => `N${x}${n(m)}${x}${m.D}`, cls: m => `N${x}${m.D}`,
    mom: () => 'EMA copy:|no gradient', txt: () => 'N strings', tok: () => `N${x}L${x}${T}`,
    te: () => `N${x}L${x}${T}`, ie: () => `N'${x}L${x}${T}`, dec: () => `N${x}L${x}${T}`,
    sim: () => `N${x}N (+ queue)`, itm: () => `N'${x}2`, lmh: () => `N${x}L${x}vocab`,
    litc: () => 'one number', litm: () => 'one number', llm: () => 'one number', loss: () => 'sum of three',
  };
  const SUB = {
    mom: () => 'soft targets', cls: m => `[CLS]: N${x}${m.D}`, vit: m => `${m.layers} layers`,
    ie: () => `N'${x}L${x}${T}|pos + hard neg`, te: () => `N${x}L${x}${T}|no image`,
  };
  const PARAMS = {
    pat: 'patch emb,|pos, [CLS]', vit: 'all ViT layers', tok: 'word emb|(shared)',
    te: 'SA-bi, FFN|(shared)', ie: 'SA-bi, CA,|FFN (shared)', dec: 'own causal SA|shared CA, FFN',
    itm: 'ITM linear', lmh: 'output layer',
  };
  ArchFlow.mount({
    prefix: 'blipflow', grid: { cols: 7, rows: 4 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['photos'], glyph: 'photos', data: true },
      pat: { c: 1, r: 0, lane: 'blue', title: ['patches', '+ [CLS]'], glyph: 'grid' },
      vit: { c: 2, r: 0, lane: 'blue', title: ['ViT', '(one pass)'], glyph: 'layers' },
      cls: { c: 3, r: 0, lane: 'blue', title: ['image', '[CLS] out'], glyph: 'vector' },
      mom: { c: 4, r: 0, lane: 'mixed', title: ['momentum', 'encoders'], glyph: 'layers' },
      txt: { c: 0, r: 2, lane: 'yellow', title: ['captions'], glyph: 'text', data: true },
      tok: { c: 1, r: 2, lane: 'yellow', title: ['tokens', '+ embed'], glyph: 'tokens' },
      te: { c: 3, r: 1, lane: 'yellow', title: ['text', 'encoder'], glyph: 'layers' },
      ie: { c: 3, r: 2, lane: 'green', title: ['grounded', 'encoder'], glyph: 'layers' },
      dec: { c: 3, r: 3, lane: 'green', title: ['grounded', 'decoder'], glyph: 'layers' },
      sim: { c: 4, r: 1, lane: 'green', title: ['similarity'], glyph: 'matrix' },
      itm: { c: 4, r: 2, lane: 'green', title: ['ITM head'], glyph: 'vector' },
      lmh: { c: 4, r: 3, lane: 'yellow', title: ['next-word', 'logits'], glyph: 'tokens' },
      litc: { c: 5, r: 1, lane: 'red', title: ['ITC loss'], glyph: 'loss' },
      litm: { c: 5, r: 2, lane: 'red', title: ['ITM loss'], glyph: 'loss' },
      llm: { c: 5, r: 3, lane: 'red', title: ['LM loss'], glyph: 'loss' },
      loss: { c: 6, r: 2, lane: 'red', title: ['total', 'loss'], glyph: 'loss' },
    },
    edges: [['img', 'pat'], ['pat', 'vit'], ['vit', 'cls'], ['cls', 'sim'], ['vit', 'ie'], ['vit', 'dec'],
      ['txt', 'tok'], ['tok', 'te'], ['tok', 'ie'], ['tok', 'dec'], ['te', 'sim'], ['ie', 'itm'], ['dec', 'lmh'],
      ['sim', 'litc'], ['mom', 'litc'], ['itm', 'litm'], ['lmh', 'llm'], ['litc', 'loss'], ['litm', 'loss'], ['llm', 'loss']],
    stages: [
      { key: 'batch', show: ['img', 'txt'] }, { key: 'tokens', show: ['pat', 'tok'] }, { key: 'vit', show: ['vit', 'cls'] },
      { key: 'itc', show: ['te', 'sim', 'mom'] }, { key: 'itm', show: ['ie', 'itm'] }, { key: 'lm', show: ['dec', 'lmh'] },
      { key: 'loss', show: ['litc', 'litm', 'llm', 'loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'b',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => SUB[id] ? SUB[id](m) : null,
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => {
      const g = R / m.P;
      return {
        overview: `${m.name}: a ${R} ${x} ${R} crop becomes ${g * g} patch tokens plus [CLS], ${n(m)} ${x} ${m.D}; the caption becomes L tokens of width ${T}. One ViT pass, three text passes, three losses.`,
        batch: `Step 1: a batch of N image-caption pairs, images ${SH.img()}; captions are N strings. BLIP uses N = ${m.batch} for ${m.name}.`,
        tokens: `Step 2: image cut into ${g} ${x} ${g} = ${g * g} patches of ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers, each mapped to ${m.D}, plus [CLS]: ${SH.pat(m)}. Caption: N ${x} L token ids, embedded to ${SH.tok()}.`,
        vit: `Step 3: the ViT (${m.layers} layers, width ${m.D}) runs once per pair: ${SH.vit(m)}. The [CLS] row, ${SH.cls(m)}, goes to ITC; all ${n(m)} rows are the keys and values for cross-attention.`,
        itc: `Step 4, ITC: the caption alone through the text encoder, [CLS] + L tokens, ${SH.te()}; its [CLS] row is compared with every image's [CLS] row: an N ${x} N table within the batch (plus, per BLIP-2, a momentum queue of negatives, size not printed). The momentum encoders supply soft targets.`,
        itm: `Step 5, ITM: N' pairs (the N true pairs plus negatives sampled by ITC score), [Encode] + text, ${SH.ie()}; cross-attention reads the N' ${x} ${n(m)} ${x} ${m.D} image features. The ITM head (one linear layer) maps the [Encode] row to match / no match; two outputs is our reading.`,
        lm: `Step 6, LM: [Decode] + caption through the decoder with causal self-attention, ${SH.dec()}, cross-attending to the image; the output layer gives N ${x} L ${x} |V| next-word scores.`,
        loss: 'Step 7: ITC (soft-target contrastive), ITM (binary cross-entropy) and LM (next-word cross-entropy, label smoothing 0.1), jointly optimized: one number.',
        backward: 'Backward: the three losses flow back through their own passes and meet in the shared layers and in the ViT. Red outlines mark what gets updated; nothing in BLIP is frozen.',
        update: `Update: one AdamW step for every red-outlined weight (${m.name}: lr warmed up to ${m.lr}, batch ${m.batch}); the momentum encoders then move toward the new weights without a gradient of their own.`,
      }[s];
    },
  });
})();
</script>

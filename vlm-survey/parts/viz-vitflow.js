<script>
(() => {
  const x = '×', PT = 224;
  // ViT Table 1 (layers, D, MLP, heads) and App. B.1 (pre-training: batch 4096 at 224 px on JFT-300M; LR and epochs per model).
  // Fine-tune rows: Sec. 4.1 (512 px for L/16, 518 px for H/14; SGD, batch 512; ImageNet, 1k classes).
  const M = {
    b16: { name: 'ViT-B/16', P: 16, L: 12, D: 768, mlp: 3072, h: 12, lr: '8e-4', ep: [7] },
    l16: { name: 'ViT-L/16', P: 16, L: 24, D: 1024, mlp: 4096, h: 16, lr: '4e-4', ep: [7, 14] },
    h14: { name: 'ViT-H/14', P: 14, L: 32, D: 1280, mlp: 5120, h: 16, lr: '3e-4', ep: [14] },
  };
  const V = {
    b16: { ...M.b16, ft: false, r: PT, bs: 4096 }, l16: { ...M.l16, ft: false, r: PT, bs: 4096 }, h14: { ...M.h14, ft: false, r: PT, bs: 4096 },
    l16ft: { ...M.l16, ft: true, r: 512, bs: 512 }, h14ft: { ...M.h14, ft: true, r: 518, bs: 512 },
  };
  const n = m => (m.r / m.P) ** 2, n0 = m => (PT / m.P) ** 2, K = m => m.ft ? '1k' : '18k';
  const fmt = v => Math.round(v).toLocaleString('en-US');
  const steps = e => { const v = 303e6 * e / 4096; return v >= 1e6 ? (v / 1e6).toFixed(2) + ' million' : fmt(v / 1000) + 'k'; };
  const SH = {
    i0: m => `${m.bs}${x}3${x}${m.r}${x}${m.r}`, i1: m => `${m.bs}${x}${n(m)}${x}${m.P * m.P * 3}`, i2: m => `${m.bs}${x}${n(m) + 1}${x}${m.D}`,
    i3: m => `${m.bs}${x}${n(m) + 1}${x}${m.D}`, i4: m => `${m.bs}${x}${m.D}`, i5: m => `${m.bs}${x}${K(m)}`, loss: () => 'one number' };
  ArchFlow.mount({
    prefix: 'vitflow', grid: { cols: 7, rows: 1 },
    boxes: {
      i0: { c: 0, r: 0, lane: 'blue', title: ['images'], glyph: 'photos', data: true },
      i1: { c: 1, r: 0, lane: 'blue', title: ['patches,', 'flattened'], glyph: 'grid', cls: false },
      i2: { c: 2, r: 0, lane: 'blue', title: ['× E, + [class]', '+ E_pos'], glyph: 'grid' },
      i3: { c: 3, r: 0, lane: 'blue', title: ['L pre-norm', 'blocks'], glyph: 'layers' },
      i4: { c: 4, r: 0, lane: 'blue', title: ['[class] out', '+ LN'], glyph: 'vector' },
      i5: { c: 5, r: 0, lane: 'blue', title: ['head'], glyph: 'mlp' },
      loss: { c: 6, r: 0, lane: 'red', title: ['loss vs', 'class labels'], glyph: 'loss' },
    },
    edges: [['i0', 'i1'], ['i1', 'i2'], ['i2', 'i3'], ['i3', 'i4'], ['i4', 'i5'], ['i5', 'loss']],
    stages: [
      { key: 'batch', show: ['i0'] }, { key: 'patches', show: ['i1'] }, { key: 'embed', show: ['i2'] }, { key: 'encode', show: ['i3'] },
      { key: 'pool', show: ['i4'] }, { key: 'head', show: ['i5'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'b16',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'i3' ? `${m.L} layers|${SH.i3(m)}` : id === 'i2' && m.ft ? `pos ${n0(m) + 1}→${n(m) + 1} rows|${SH.i2(m)}` : id === 'i5' ? (m.ft ? `new, zero-init|${SH.i5(m)}` : `MLP, 1 hidden|${SH.i5(m)}`) : null,
    params: (id, m) => ({ i2: 'E, [class], E_pos', i3: 'all blocks', i4: 'LN', i5: m.ft ? 'D×K, from zero' : 'head MLP' })[id] || null,
    frozen: () => false,
    readout: (s, m) => {
      const g = m.r / m.P, g0 = PT / m.P, att = (n(m) + 1) ** 2;
      return {
        overview: `${m.name}, ${m.ft ? 'fine-tuned on ImageNet' : 'pre-trained on JFT-300M'} at ${m.r} px: ${fmt(n(m))} patch tokens plus [class], width ${m.D}, ${m.L} blocks. One lane only: an image in, a class score out.`,
        batch: `Step 1: ${m.ft ? 'a fine-tuning' : 'a pre-training'} batch of ${m.bs} images at ${m.r} ${x} ${m.r}: ${SH.i0(m)}, each with its label(s).`,
        patches: `Step 2: a ${g} ${x} ${g} grid of ${m.P} ${x} ${m.P} patches, each flattened to ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers: ${SH.i1(m)}.`,
        embed: `Step 3: E (${m.P * m.P * 3} ${x} ${m.D}) maps every patch to ${m.D} numbers; prepend [class]; add E_pos (${n(m) + 1} ${x} ${m.D}): ${SH.i2(m)}.` +
          (m.ft ? ` E_pos was learned for a ${g0} ${x} ${g0} grid (${n0(m) + 1} rows); it is 2D-interpolated to ${g} ${x} ${g} (${n(m) + 1} rows).` : ''),
        encode: `Step 4: ${m.L} blocks, each LN, ${m.h}-head self-attention (head width ${m.D} / ${m.h} = ${m.D / m.h}), residual, LN, MLP ${m.D}→${m.mlp}→${m.D}, residual. Shape stays ${SH.i3(m)}; each attention map is ${fmt(n(m) + 1)} ${x} ${fmt(n(m) + 1)} = ${fmt(att)} pairs per head.`,
        pool: `Step 5: keep only token 0, the [class] output, and layer-normalize it (Eq. 4): ${SH.i4(m)}. The other ${fmt(n(m))} outputs are dropped.`,
        head: m.ft ? `Step 6: the pre-training head is gone; a new ${m.D} ${x} K linear layer, zero-initialized, gives ${SH.i5(m)} (K = 1k ImageNet classes).`
          : `Step 6: an MLP with one hidden layer (tanh; width not stated) maps each image to one score per class: ${SH.i5(m)} (K is about 18k JFT classes).`,
        loss: 'Step 7: the scores are graded against the label(s), one image at a time, and averaged over the batch. Only the [class] position is graded; the paper does not name the loss function.',
        backward: `Backward: from the loss into the head, into the [class] output, then down all ${m.L} blocks. The patch tokens' outputs are never graded; they get gradient only by way of attention, because [class] read them, directly or through other patches.` +
          (m.ft ? ' At the very first step the head is all zeros, so no gradient reaches the backbone until the head has moved (derived).' : ''),
        update: m.ft ? `Update: SGD, momentum 0.9, batch 512, cosine decay, no weight decay, gradient clipping at norm 1; ImageNet runs 20,000 steps. ${m.name} at ${m.r} px is a Table 2 model, which also keeps a Polyak average of the weights.`
          : `Update: Adam (0.9, 0.999), batch 4096, weight decay 0.1, 10k warm-up steps then linear decay from ${m.lr}; ${m.ep.join(' or ')} epochs of JFT (303M images), about ${m.ep.map(e => steps(e)).join(' or ')} steps.`,
      }[s];
    },
  });
})();
</script>

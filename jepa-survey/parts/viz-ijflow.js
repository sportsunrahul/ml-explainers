<script>
(() => {
  const x = '×';
  // Encoder sizes from the released vision_transformer.py (code-only); predictor from Appendix A.
  const V = {
    b16: { name: 'ViT-B/16', P: 16, r: 224, D: 768, L: 12, H: 12, pd: 6, ep: 600 },
    l16: { name: 'ViT-L/16', P: 16, r: 224, D: 1024, L: 24, H: 16, pd: 12, ep: 600 },
    h14: { name: 'ViT-H/14', P: 14, r: 224, D: 1280, L: 32, H: 16, pd: 12, ep: 300 },
    h16: { name: 'ViT-H/16 at 448', P: 16, r: 448, D: 1280, L: 32, H: 16, pd: 12, ep: 300 },
  };
  const n = m => (m.r / m.P) ** 2, nc = m => Math.round(.25 * n(m)), nt = m => [Math.floor(.15 * n(m)), Math.floor(.2 * n(m))];
  const SH = {
    img: m => `B${x}3${x}${m.r}${x}${m.r}`, pat: m => `B${x}${n(m)}${x}${m.D}`, tmask: m => `4 blocks of ${nt(m)[0]}-${nt(m)[1]} patches`, cmask: m => `~${nc(m)} context patches`,
    tenc: m => `B${x}${n(m)}${x}${m.D}`, cenc: m => `B${x}~${nc(m)}${x}${m.D}`, pred: m => `B${x}(~${nc(m)}+n_i)${x}384 -> B${x}n_i${x}${m.D}`, loss: () => 'one number' };
  const PARAMS = { pat: 'patch matrix', cenc: 'all layers', pred: 'in/out projections, mask token, all layers' };
  ArchFlow.mount({
    prefix: 'ijflow', grid: { cols: 6, rows: 2 },
    boxes: {
      img: { c: 0, r: .5, lane: 'blue', title: ['image'], glyph: 'photos', data: true },
      pat: { c: 1, r: .5, lane: 'blue', title: ['patches', '+ position'], glyph: 'grid' },
      tmask: { c: 2, r: 0, lane: 'yellow', title: ['target', 'blocks'], glyph: 'grid' },
      cmask: { c: 2, r: 1, lane: 'blue', title: ['context', 'block'], glyph: 'grid' },
      tenc: { c: 3, r: 0, lane: 'yellow', title: ['target encoder', '(EMA copy)'], glyph: 'layers' },
      cenc: { c: 3, r: 1, lane: 'blue', title: ['context', 'encoder'], glyph: 'layers' },
      pred: { c: 4, r: 1, lane: 'green', title: ['predictor', '+ mask tokens'], glyph: 'layers' },
      loss: { c: 5, r: .5, lane: 'red', title: ['L2 loss'], glyph: 'loss' },
    },
    edges: [['img', 'pat'], ['pat', 'tmask'], ['pat', 'cmask'], ['tmask', 'tenc'], ['cmask', 'cenc'], ['cenc', 'pred'], ['pred', 'loss'], ['tenc', 'loss']],
    stages: [
      { key: 'image', show: ['img'] }, { key: 'patches', show: ['pat'] }, { key: 'masks', show: ['tmask', 'cmask'] }, { key: 'encoders', show: ['tenc', 'cenc'] },
      { key: 'predictor', show: ['pred'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'h14',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'tenc' || id === 'cenc' ? `${m.L} layers, ${m.H} heads, width ${m.D}` : id === 'pred' ? `${m.pd} layers, ${m.H} heads, width 384` : null,
    params: id => PARAMS[id] || null,
    frozen: id => id === 'tenc',
    readout: (s, m) => {
      const g = m.r / m.P;
      return {
        overview: `${m.name}: a ${m.r} ${x} ${m.r} image becomes ${g} ${x} ${g} = ${n(m)} patch tokens of width ${m.D}. The target encoder sees all ${n(m)}; the context encoder about ${nc(m)}.`,
        image: `Step 1: B images, ${SH.img(m)}. One view each; no augmentation beyond a random resized crop.`,
        patches: `Step 2: ${n(m)} patches of ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers, each mapped to ${m.D}, plus 2D sin-cos positions: ${SH.pat(m)}.`,
        masks: `Step 3: four target blocks of ${nt(m)[0]}-${nt(m)[1]} patches each (15-20% of ${n(m)}); one square context block of 85-100%, minus the target patches: about ${nc(m)} remain (the paper's 0.25 average, measured on ViT-B/16).`,
        encoders: `Step 4: target encoder on all ${n(m)} tokens (${SH.tenc(m)}, no gradient, EMA weights); context encoder on ~${nc(m)} tokens (${SH.cenc(m)}). Same architecture: ${m.L} layers, width ${m.D}, ${m.H} heads.`,
        predictor: `Step 5: project ~${nc(m)} context vectors to width 384, append n_i mask tokens (n_i in ${nt(m)[0]}..${nt(m)[1]}), run ${m.pd} layers, project the n_i outputs back to ${m.D}. Four passes, one per block.`,
        loss: `Step 6: for each block, sum over its n_i patches of the squared distance between the ${m.D}-vectors; average over the 4 blocks. One number.`,
        backward: 'Backward: the gradient leaves the loss into the predictor (red) and on into the context encoder (red). The target encoder is brown: its outputs are constants and nothing trainable sits behind it, so the gradient stops.',
        update: `Update: AdamW steps the context encoder and predictor; then the target encoder moves toward the new context encoder by (1 - m), with m from 0.996 to 1.0. ${m.ep} epochs at batch 2048.`,
      }[s];
    },
  });
})();
</script>

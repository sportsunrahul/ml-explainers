<script>
(() => {
  const x = '×';
  const V = { tf: { name: 'teacher forcing', T: 15 }, roll: { name: 'rollout', T: 2 } };
  const SH = { clip: () => `B${x}16${x}3${x}256${x}256 + 16 states (7)`, enc: () => `16 maps of 16${x}16${x}1408 (frozen)`, tok: m => `per step: 1 action + 1 state + 256 patches, width 1024; ${m.T} steps`, pred: m => `${m.T} predicted maps 16${x}16${x}1408`, loss: () => 'one number' };
  ArchFlow.mount({
    prefix: 'acflow', grid: { cols: 5, rows: 2 },
    boxes: {
      clip: { c: 0, r: .5, lane: 'blue', title: ['robot clip', '+ states'], glyph: 'photos', data: true },
      enc: { c: 1, r: 0, lane: 'blue', title: ['V-JEPA 2', 'encoder'], glyph: 'layers' },
      tok: { c: 1, r: 1, lane: 'yellow', title: ['action, state', 'tokens'], glyph: 'tokens' },
      pred: { c: 2, r: .5, lane: 'green', title: ['action-conditioned', 'predictor'], glyph: 'layers' },
      tgt: { c: 3, r: 0, lane: 'yellow', title: ['next-frame', 'features'], glyph: 'vector' },
      loss: { c: 4, r: .5, lane: 'red', title: ['L1 loss'], glyph: 'loss' },
    },
    edges: [['clip', 'enc'], ['clip', 'tok'], ['enc', 'pred'], ['tok', 'pred'], ['enc', 'tgt'], ['pred', 'loss'], ['tgt', 'loss']],
    stages: [{ key: 'clip', show: ['clip'] }, { key: 'encode', show: ['enc'] }, { key: 'tokens', show: ['tok'] }, { key: 'predict', show: ['pred'] }, { key: 'loss', show: ['tgt', 'loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'tf',
    shape: (id, m) => ({ clip: SH.clip(), enc: SH.enc(), tok: SH.tok(m), pred: SH.pred(m), tgt: `${m.T} maps 16${x}16${x}1408`, loss: SH.loss() })[id],
    sub: (id, m) => id === 'enc' ? 'ViT-g, 1B, frozen' : id === 'pred' ? '24 layers, 16 heads, block-causal' : null,
    params: id => id === 'pred' ? 'input maps, 24 layers, output map' : null,
    frozen: id => id === 'enc' || id === 'tgt',
    readout: (s, m) => ({
      overview: `${m.name}: the frozen encoder turns every frame into a 16 x 16 x 1408 map; the predictor reads ${m.T} steps of (action, state, map) and predicts the next map.`,
      clip: 'Step 1: 16 frames at 4 fps, 256 pixels, with the 7D end-effector state per frame; actions are the per-step state deltas.',
      encode: 'Step 2: each frame independently through the frozen ViT-g: 256 patch features of width 1408 per frame. Brown: no gradient, ever.',
      tokens: `Step 3: actions and states each get a linear map to width 1024; so do the patch features. ${m.T} time steps of (1 + 1 + 256) tokens, time positions by RoPE.`,
      predict: m.T === 15 ? 'Step 4 (teacher forcing): true maps as inputs at every step; block-causal attention; output at step k is the prediction for frame k+1.' : 'Step 4 (rollout): from frame 1, the predictor consumes its own predicted map for the next step; two steps, gradient through one.',
      loss: 'Step 5: L1 between each predicted map and the frozen encoder\'s map of the real next frame; the two stages\' losses are added.',
      backward: 'Backward: the gradient leaves the loss into the predictor only. Both encoder outputs are constants, so the gradient stops at the predictor\'s input maps.',
      update: 'Update: AdamW on the predictor. Batch 256; warm-up 4500 steps to 4.25e-4, 85500 constant, then decay.',
    })[s],
  });
})();
</script>

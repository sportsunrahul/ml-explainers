<script>
(() => {
  const x = '×';
  // Input sizes, block indices, feature sizes and step counts: FastFlow v2 Table 6 and Sec. 4.7. Channel widths C are the
  // standard backbone widths (our reading; the paper does not state them), so the figure prints them as "C".
  const V = {
    cait: { name: 'CaiT-M48', r: 448, maps: [28], C: ['768'], steps: 20, k: '3x3 and 1x1 alternating', flows: 1 },
    deit: { name: 'DeiT-base', r: 384, maps: [24], C: ['768'], steps: 20, k: '3x3 and 1x1 alternating', flows: 1 },
    r18: { name: 'ResNet18', r: 256, maps: [64, 32, 16], C: ['64', '128', '256'], steps: 8, k: '3x3 only', flows: 3 },
    wrn50: { name: 'WRN50-2', r: 256, maps: [64, 32, 16], C: ['256', '512', '1024'], steps: 8, k: '3x3 and 1x1 alternating', flows: 3 },
  };
  const maps = m => m.maps.map(h => `C${x}${h}${x}${h}`).join(', ');
  ArchFlow.mount({
    prefix: 'ffflow', grid: { cols: 5, rows: 2 },
    boxes: {
      img: { c: 0, r: .5, lane: 'blue', title: ['good image'], glyph: 'photo', data: true },
      bb: { c: 1, r: .5, lane: 'blue', title: ['backbone', '(frozen)'], glyph: 'layers' },
      flow: { c: 2, r: .5, lane: 'green', title: ['2D flow'], glyph: 'layers' },
      z: { c: 3, r: 0, lane: 'green', title: ['latent z'], glyph: 'grid' },
      base: { c: 3, r: 1, lane: 'blue', title: ['N(0, I)'], glyph: 'unit', data: true },
      loss: { c: 4, r: .5, lane: 'red', title: ['-log p,', 'every position'], glyph: 'loss' },
    },
    edges: [['img', 'bb'], ['bb', 'flow'], ['flow', 'z'], ['z', 'loss'], ['base', 'loss'], ['flow', 'loss']],
    stages: [
      { key: 'image', show: ['img'] }, { key: 'backbone', show: ['bb'] }, { key: 'flow', show: ['flow'] }, { key: 'latent', show: ['z', 'base'] },
      { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'cait',
    shape: (id, m) => ({ img: `3${x}${m.r}${x}${m.r}`, bb: maps(m), flow: maps(m), z: maps(m), base: 'same shape', loss: 'one number' })[id],
    sub: (id, m) => id === 'flow' ? `${m.steps} steps${m.flows > 1 ? ` ${x} ${m.flows} flows` : ''}, ${m.k}` : id === 'bb' ? `C = ${m.C.join(', ')} (our reading)` : null,
    params: id => id === 'flow' ? 'ActNorm, conv subnets s and b' : null,
    frozen: id => id === 'bb',
    readout: (s, m) => ({
      overview: `${m.name}: a ${m.r} ${x} ${m.r} image gives ${m.maps.length > 1 ? 'three feature maps' : 'one feature map'} of ${m.maps.join(', ')} ${x} ${m.maps.join(', ')} positions. ${m.flows > 1 ? 'Three separate flows, one per map.' : 'One flow.'}`,
      image: `Step 1: a good image, resized to ${m.r} ${x} ${m.r}. Training uses normal images only.`,
      backbone: `Step 2: the frozen, ImageNet-pretrained backbone gives ${maps(m)}. The paper states the spatial sizes; C is the standard width (our reading).`,
      flow: `Step 3: ${m.steps} flow steps${m.flows > 1 ? ' per map' : ''}. Each step keeps the shape ${m.maps.length > 1 ? 'of its map' : maps(m)}: an invertible map from features to latents of the same size.`,
      latent: `Step 4: the latent z has the shape of the feature map. Training pushes every position of z toward a standard normal, N(0, I).`,
      loss: 'Step 5: at every position, -log p = -log N(z) - log|det|; averaged over positions and images (our reading of "maximize the log-likelihood").',
      backward: 'Backward: the gradient leaves the loss into the flow (red) through both terms, z and the log-det. The backbone is brown: frozen, and nothing trainable sits upstream of it, so the gradient stops at its output.',
      update: `Update: Adam steps the flow only. ${m.flows > 1 ? 'Each of the three flows learns its own scale.' : ''} At test time the same pass gives a likelihood map; no gradient is needed.`,
    })[s],
  });
})();
</script>

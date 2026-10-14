<script>
(() => {
  const x = '×';
  // All shapes are code-only (official DRAEM repo, model_unet.py and train_DRAEM.py, input 256): the paper states neither
  // the resolution nor the widths. Reconstructive subnet: base width 128, bottleneck 1024 x 16 x 16, no skips.
  // Discriminative subnet: U-Net, base width 64, 6 input channels (reconstruction + input), 2 output channels.
  ArchFlow.mount({
    prefix: 'draemflow', grid: { cols: 6, rows: 2 },
    boxes: {
      img: { c: 0, r: .5, lane: 'blue', title: ['good image I'], glyph: 'photo', data: true },
      sim: { c: 1, r: .5, lane: 'yellow', title: ['fake anomaly', 'I_a and mask M'], glyph: 'photo' },
      rec: { c: 2, r: 0, lane: 'green', title: ['reconstructive', 'subnet'], glyph: 'layers' },
      cat: { c: 3, r: 1, lane: 'mixed', title: ['concat', 'I_r and I_a'], glyph: 'grid' },
      disc: { c: 4, r: 1, lane: 'green', title: ['discriminative', 'subnet (U-Net)'], glyph: 'layers' },
      lrec: { c: 3, r: 0, lane: 'red', title: ['L_rec: SSIM + L2', 'vs clean I'], glyph: 'loss' },
      lseg: { c: 5, r: 1, lane: 'red', title: ['focal loss', 'vs mask M'], glyph: 'loss' },
    },
    edges: [['img', 'sim'], ['sim', 'rec'], ['rec', 'cat'], ['sim', 'cat'], ['cat', 'disc'], ['disc', 'lseg'], ['rec', 'lrec']],
    stages: [
      { key: 'image', show: ['img'] }, { key: 'simulate', show: ['sim'] }, { key: 'reconstruct', show: ['rec'] }, { key: 'compare', show: ['cat', 'disc'] },
      { key: 'losses', show: ['lrec', 'lseg'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: {}, variant: null,
    shape: id => ({ img: `3${x}256${x}256`, sim: `3${x}256${x}256 + mask`, rec: `3${x}256${x}256`, cat: `6${x}256${x}256`, disc: `2${x}256${x}256`, lrec: 'one number', lseg: 'one number' })[id],
    sub: id => id === 'rec' ? `bottleneck 1024${x}16${x}16, no skips` : id === 'disc' ? 'base width 64, skips' : id === 'sim' ? 'Perlin mask, DTD texture' : null,
    params: id => id === 'rec' ? 'encoder-decoder convs' : id === 'disc' ? 'U-Net convs' : null,
    frozen: () => false,
    readout: s => ({
      overview: 'Two networks, both trained from scratch, nothing pretrained: one repairs the fake, the other compares the repair with the input and segments the difference.',
      image: `Step 1: a good image, 3${x}256${x}256 in the official code (the paper does not state the resolution). Half of the training samples stay clean (code-only).`,
      simulate: 'Step 2: a Perlin-noise mask and a texture from an unrelated dataset, blended with opacity beta (Eq. 4). Output: the fake I_a and its mask M.',
      reconstruct: `Step 3: the reconstructive subnet maps I_a back to a clean-looking image I_r: down to 1024${x}16${x}16 and up again, with no skip connections, so it cannot simply copy the fake through.`,
      compare: `Step 4: I_r and I_a are stacked into 6 channels; the U-Net outputs 2 channels per pixel, defect or not. Its softmax is the anomaly map M_o.`,
      losses: 'Step 5: L_rec compares I_r with the clean I (SSIM plus L2); the focal loss compares the map with the fake mask M.',
      backward: 'Backward: the focal loss reaches the U-Net, and through I_r it also reaches the reconstructive subnet (the code does not detach it). L_rec reaches the reconstructive subnet directly. Nothing is frozen.',
      update: 'Update: one optimizer steps both subnets together on the sum of the two losses. At test time only the forward pass runs: the image score is the maximum of the smoothed map (Eq. 5).',
    })[s],
  });
})();
</script>

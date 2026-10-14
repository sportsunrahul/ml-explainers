<script>
(() => {
  // PaDiM's fit-and-score pipeline. Stated: 224 input, layers 1-3 (ResNets), 448 dims for ResNet18, 100 / 550 kept, eps.
  // Our reading: channel widths (standard torchvision), the 56 x 56 grid, 1792 dims for WR50, storage per position.
  const V = { r18: { name: 'ResNet18, 100 dims', ch: [64, 128, 256], d: 100 }, wr50: { name: 'Wide-ResNet-50-2, 550 dims', ch: [256, 512, 1024], d: 550 } };
  const sum = v => v.ch.reduce((a, b) => a + b, 0), store = v => v.d + v.d * v.d;
  const fmt = n => n.toLocaleString('en-US');
  ArchFlow.mount({
    prefix: 'padimflow', grid: { cols: 6, rows: 2 },
    boxes: {
      imgs: { c: 0, r: 0, lane: 'blue', title: ['normal', 'images'], glyph: 'photos', data: true },
      test: { c: 0, r: 1, lane: 'yellow', title: ['test', 'image'], glyph: 'photo', data: true },
      cnn: { c: 1, r: .5, lane: 'mixed', title: ['pretrained CNN,', 'layers 1-3'], glyph: 'layers' },
      cat: { c: 2, r: .5, lane: 'mixed', title: ['join on the', 'finest grid'], glyph: 'grid', cls: false },
      rd: { c: 3, r: .5, lane: 'mixed', title: ['random', 'dimensions'], glyph: 'vector' },
      gauss: { c: 4, r: 0, lane: 'blue', title: ['one Gaussian', 'per position'], glyph: 'grid', cls: false, data: true },
      maha: { c: 4, r: 1, lane: 'red', title: ['Mahalanobis', 'per position'], glyph: 'grid', cls: false },
      map: { c: 5, r: 1, lane: 'red', title: ['anomaly map,', 'max = score'], glyph: 'loss' },
    },
    edges: [['imgs', 'cnn'], ['test', 'cnn'], ['cnn', 'cat'], ['cat', 'rd'], ['rd', 'gauss'], ['rd', 'maha'], ['gauss', 'maha'], ['maha', 'map']],
    stages: [
      { key: 'image', show: ['imgs', 'test'] }, { key: 'features', show: ['cnn'] }, { key: 'concat', show: ['cat'] }, { key: 'select', show: ['rd'] },
      { key: 'fit', show: ['gauss'] }, { key: 'score', show: ['maha'] }, { key: 'map', show: ['map'] }, { key: 'backward', back: true }],
    variants: V, variant: 'wr50',
    shape: (id, v) => ({ imgs: 'N x 3 x 224 x 224', test: '1 x 3 x 224 x 224', cnn: `${v.ch[0]} x 56 x 56|${v.ch[1]} x 28 x 28|${v.ch[2]} x 14 x 14`,
      cat: `${sum(v)} x 56 x 56`, rd: `${v.d} x 56 x 56`, gauss: `3,136 x (mean ${v.d},|cov ${v.d} x ${v.d})`, maha: '56 x 56 distances', map: 'resize, smooth, max' })[id],
    params: () => null,
    frozen: id => id === 'cnn',
    readout: (s, v) => ({
      overview: `PaDiM (${v.name}): a frozen CNN, one Gaussian per position of a 56 x 56 grid, Mahalanobis distances as the map.`,
      image: 'Step 1: resize to 256, crop to 224. Normal images: N x 3 x 224 x 224 at fitting time; the test image: 1 x 3 x 224 x 224.',
      features: `Step 2: layers 1-3 give ${v.ch[0]} x 56 x 56, ${v.ch[1]} x 28 x 28 and ${v.ch[2]} x 14 x 14 (standard widths: our reading).`,
      concat: `Step 3: every position of the 56 x 56 grid gets the vectors that cover it, joined: ${sum(v)} dimensions${v.d === 100 ? ' (448 is stated in the paper)' : ' (our reading)'}.`,
      select: `Step 4: keep ${v.d} randomly chosen dimensions, the same ones at every position and for every image.`,
      fit: `Step 5: per position, a ${v.d}-d mean and a ${v.d} x ${v.d} covariance plus 0.01 I: ${fmt(store(v))} numbers per position, ${(store(v) * 4 * 3136 / 1e9).toFixed(2)} GB for 3,136 positions in 32-bit floats (our computation).`,
      score: 'Step 6: at each position, the Mahalanobis distance of the test vector to that position\'s Gaussian (Eq. 2).',
      map: 'Step 7: the 56 x 56 map, resized (bicubic) and smoothed (Gaussian, width 4); the image score is its maximum.',
      backward: 'Backward: no loss and no trainable weight. The CNN is frozen and the Gaussians are computed in closed form, so no gradient flows anywhere.',
    })[s],
  });
})();
</script>

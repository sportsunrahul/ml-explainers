<script>
(() => {
  // SPADE as two paths through one frozen backbone. Sizes from Sec. 3.1-3.5 of v3; channel widths are the standard
  // Wide-ResNet-50-2 widths (our reading); the gallery size 50 x 3,136 is our computation.
  const SH = {
    imgs: 'N x 3 x 224 x 224', test: '1 x 3 x 224 x 224', cnn: 'Wide-ResNet50x2', pool: 'one 2048-d vector',
    pyr: '256 x 56 x 56|512 x 28 x 28|1024 x 14 x 14', knn: 'K = 50 nearest images', iscore: 'one number per image',
    gal: '50 x 3,136 pixel vectors', pix: '56 x 56 map, sigma 4 smoothing' };
  ArchFlow.mount({
    prefix: 'spadeflow', grid: { cols: 5, rows: 2 },
    boxes: {
      imgs: { c: 0, r: 0, lane: 'blue', title: ['normal', 'images'], glyph: 'photos', data: true },
      test: { c: 0, r: 1, lane: 'yellow', title: ['test', 'image'], glyph: 'photo', data: true },
      cnn: { c: 1, r: .5, lane: 'mixed', title: ['pretrained', 'backbone'], glyph: 'layers' },
      pool: { c: 2, r: 0, lane: 'mixed', title: ['last block,', 'pooled'], glyph: 'vector' },
      pyr: { c: 2, r: 1, lane: 'mixed', title: ['blocks 1-3', '(pyramid)'], glyph: 'grid', cls: false },
      knn: { c: 3, r: 0, lane: 'blue', title: ['image', 'kNN'], glyph: 'vector' },
      iscore: { c: 4, r: 0, lane: 'red', title: ['image', 'score'], glyph: 'loss' },
      gal: { c: 3, r: 1, lane: 'blue', title: ['gallery of', 'K images'], glyph: 'grid', cls: false, data: true },
      pix: { c: 4, r: 1, lane: 'red', title: ['pixel kNN,', 'anomaly map'], glyph: 'grid', cls: false },
    },
    edges: [['imgs', 'cnn'], ['test', 'cnn'], ['cnn', 'pool'], ['cnn', 'pyr'], ['pool', 'knn'], ['knn', 'iscore'], ['knn', 'gal'], ['pyr', 'gal'], ['pyr', 'pix'], ['gal', 'pix']],
    stages: [
      { key: 'image', show: ['imgs', 'test'] }, { key: 'backbone', show: ['cnn'] }, { key: 'paths', show: ['pool', 'pyr'] },
      { key: 'retrieve', show: ['knn', 'iscore'] }, { key: 'gallery', show: ['gal'] }, { key: 'pixel', show: ['pix'] },
      { key: 'backward', back: true }],
    variants: {}, variant: null,
    shape: id => SH[id],
    params: () => null,
    frozen: id => id === 'cnn' || id === 'pool' || id === 'pyr',
    readout: s => ({
      overview: 'SPADE: one frozen backbone, an image path (pooled vector, K nearest images, image score) and a pixel path (feature pyramid, gallery, anomaly map).',
      image: 'Step 1: resize to 256, crop to 224. Normal images: N x 3 x 224 x 224, run once at fitting time. Test image: 1 x 3 x 224 x 224.',
      backbone: 'Step 2: Wide-ResNet50x2 with ImageNet weights. It is never updated.',
      paths: 'Step 3: image path, last block averaged to one 2048-d vector. Pixel path, blocks 1-3 at 56, 28 and 14 positions per side (widths 256, 512, 1024: our reading).',
      retrieve: 'Step 4: Euclidean distance between pooled vectors; the K = 50 nearest normal images. Image score = their mean squared distance (Eq. 2).',
      gallery: 'Step 5: the pixel features of the 50 retrieved images: 50 x 3,136 = 156,800 vectors at 56 x 56 (our computation).',
      pixel: 'Step 6: per test pixel, squared distance to the single nearest gallery vector (kappa = 1, Eq. 3); Gaussian smoothing, sigma 4.',
      backward: 'Backward: no loss, no trainable weight, no gradient anywhere. Fitting SPADE means storing the features of the normal images.',
    })[s],
  });
})();
</script>

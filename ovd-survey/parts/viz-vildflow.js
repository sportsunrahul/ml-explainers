<script>
(() => {
  // ViLD (ResNet-50, CLIP ViT-B/32 teacher), Sec. 3 and Fig. 3; config from Sec. 4. N, M and the width d are
  // not stated in the paper, so they stay symbolic. The CLIP encoders ran offline: their outputs enter as data.
  const x = ' x ';
  const SH = {
    img: '1024' + x + '1024' + x + '3', bb: 'feature pyramid|(random init)', rpn: 'N boxes + scores', lrpn: 'objectness|+ deltas',
    roi: 'N + M crops', head: '(N + M)' + x + 'd', ebg: '1' + x + 'd', temb: '866' + x + 'd (train)|1,203' + x + 'd (test)',
    ltext: 'N' + x + '867 cosines|/ 0.01, softmax', crops: 'M per image,|1x and 1.5x crops', cemb: 'M' + x + 'd,|unit length',
    limg: 'mean L1|over M', ltot: 'L_text + 0.5 L_image',
  };
  const PARAMS = { bb: 'all layers,|from scratch', rpn: 'conv, objectness|and delta layers', head: 'FC layers|+ projection', ebg: 'e_bg' };
  ArchFlow.mount({
    prefix: 'vildflow', grid: { cols: 8, rows: 3 },
    boxes: {
      rpn: { c: 2, r: 0, lane: 'blue', title: ['RPN'], glyph: 'grid', cls: false },
      lrpn: { c: 3, r: 0, lane: 'red', title: ['RPN losses'], glyph: 'loss' },
      ebg: { c: 5, r: 0, lane: 'mixed', title: ['background', 'row e_bg'], glyph: 'vector' },
      temb: { c: 4, r: 0, lane: 'yellow', title: ['CLIP text', 'embeddings'], glyph: 'text', data: true },
      img: { c: 0, r: 1, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      bb: { c: 1, r: 1, lane: 'blue', title: ['ResNet-50', '+ FPN'], glyph: 'layers' },
      roi: { c: 3, r: 1, lane: 'blue', title: ['RoIAlign'], glyph: 'grid', cls: false },
      head: { c: 4, r: 1, lane: 'blue', title: ['region head', '+ projection'], glyph: 'mlp' },
      ltext: { c: 6, r: 1, lane: 'red', title: ['ViLD-text', 'loss'], glyph: 'loss' },
      ltot: { c: 7, r: 1, lane: 'red', title: ['total loss'], glyph: 'loss' },
      crops: { c: 3, r: 2, lane: 'blue', title: ['offline boxes'], glyph: 'photos', data: true },
      cemb: { c: 4, r: 2, lane: 'blue', title: ['CLIP crop', 'embeddings'], glyph: 'unit', data: true },
      limg: { c: 6, r: 2, lane: 'red', title: ['ViLD-image', 'L1 loss'], glyph: 'loss' },
    },
    edges: [['img', 'bb'], ['bb', 'rpn'], ['bb', 'roi'], ['rpn', 'roi'], ['rpn', 'lrpn'], ['crops', 'roi'], ['crops', 'cemb'], ['roi', 'head'],
      ['head', 'ltext'], ['ebg', 'ltext'], ['temb', 'ltext'], ['head', 'limg'], ['cemb', 'limg'], ['ltext', 'ltot'], ['limg', 'ltot']],
    stages: [
      { key: 'offline', show: ['temb', 'crops', 'cemb'] }, { key: 'backbone', show: ['img', 'bb'] }, { key: 'proposals', show: ['rpn', 'lrpn'] },
      { key: 'regions', show: ['roi', 'head'] }, { key: 'textloss', show: ['ebg', 'ltext'] }, { key: 'imageloss', show: ['limg'] },
      { key: 'total', show: ['ltot'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: {}, variant: null,
    shape: id => SH[id],
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: s => ({
      overview: 'ResNet-50 ViLD with a CLIP ViT-B/32 teacher. Per image: a 1024 x 1024 x 3 input, N online proposals, M stored boxes; embedding width d (N, M and d are not stated).',
      offline: 'Stored before training: 866 x d class text embeddings; per training image, M boxes and M x d CLIP crop embeddings of unit length.',
      backbone: '1024 x 1024 x 3 image -> feature pyramid: several grids at different strides.',
      proposals: 'Per image: N boxes x 4 coordinates, each with an objectness score.',
      regions: 'N online + M offline boxes -> RoIAlign crops -> head -> projection -> (N + M) x d region embeddings.',
      textloss: '(N x d) against (867 x d): N x 867 cosines, / 0.01, softmax, cross-entropy, mean over N.',
      imageloss: '(M x d) student embeddings against (M x d) stored targets: L1 distance, mean over M.',
      total: 'L = L_text + 0.5 L_image, averaged over a batch of 256 images.',
      backward: 'Gradient reaches: backbone, RPN, head and projection, e_bg. No gradient: stored text embeddings, stored crop embeddings, offline boxes.',
      update: 'Batch 256, initial learning rate 0.32, weight decay 4e-5, 180,000 iterations from random initialization.',
    })[s],
  });
})();
</script>

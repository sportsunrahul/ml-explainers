<script>
(() => {
  // AnomalyCLIP training step. Config from AnomalyCLIP v12: E = 12 learned word embeddings per prompt, 4 deep tokens in each of
  // the first 9 text layers, ViT-L/14@336px at 518 px, batch 8, lambda 4. The 37 x 37 grid is our computation (518 / 14).
  const x = '×', gsz = Math.floor(518 / 14);
  const SH = {
    prm: `2 prompts: 12 learned + fixed words`, deep: `4 tokens ${x} 9 layers`, tenc: `frozen`, proto: `g_n, g_a`,
    img: `8${x}3${x}518${x}518`, gt: `8 labels + 8 masks`, venc: `ViT-L/14@336px + DPAM`, feats: `class token + ${gsz}${x}${gsz} patches`,
    sc: `8 scores, maps ${gsz}${x}${gsz} -> 518${x}518`, loss: 'one number' };
  ArchFlow.mount({
    prefix: 'aclipflow', grid: { cols: 6, rows: 3 },
    boxes: {
      prm: { c: 0, r: 0, lane: 'green', title: ['learned', 'prompts'], glyph: 'tokens' },
      deep: { c: 0, r: 1, lane: 'green', title: ['deep text', 'tokens'], glyph: 'tokens' },
      tenc: { c: 1, r: .5, lane: 'mixed', title: ['CLIP text', 'encoder'], glyph: 'layers' },
      proto: { c: 2, r: .5, lane: 'mixed', title: ['text embeddings', 'g_n, g_a'], glyph: 'vector' },
      img: { c: 0, r: 2, lane: 'yellow', title: ['auxiliary', 'image'], glyph: 'photo', data: true },
      venc: { c: 1, r: 2, lane: 'yellow', title: ['CLIP image', 'encoder'], glyph: 'layers' },
      feats: { c: 2, r: 2, lane: 'yellow', title: ['class token +', 'patch tokens'], glyph: 'grid' },
      sc: { c: 3, r: 1.25, lane: 'red', title: ['scores and', 'maps'], glyph: 'gate' },
      gt: { c: 4, r: 2, lane: 'red', title: ['label +', 'defect mask'], glyph: 'grid', data: true },
      loss: { c: 5, r: 1.25, lane: 'red', title: ['global +', '4 x local'], glyph: 'loss' },
    },
    edges: [['prm', 'tenc'], ['deep', 'tenc'], ['tenc', 'proto'], ['proto', 'sc'], ['img', 'venc'], ['venc', 'feats'], ['feats', 'sc'], ['sc', 'loss'], ['gt', 'loss']],
    stages: [
      { key: 'prompts', show: ['prm', 'deep'] }, { key: 'text', show: ['tenc', 'proto'] }, { key: 'image', show: ['img'] }, { key: 'vision', show: ['venc', 'feats'] },
      { key: 'scores', show: ['sc'] }, { key: 'loss', show: ['gt', 'loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: {}, variant: null,
    shape: id => SH[id],
    params: id => ({ prm: 'V1..V12, W1..W12', deep: '4 tokens per layer', tenc: 'CLIP text weights', venc: 'CLIP image weights' })[id] || null,
    frozen: id => id === 'tenc' || id === 'venc',
    readout: s => ({
      overview: 'AnomalyCLIP: every CLIP weight is frozen; the only trained weights are the prompt vectors and the deep text tokens.',
      prompts: `Step 1: normal prompt = 12 learned vectors + "object"; abnormal prompt = 12 learned vectors + "damaged object". Plus ${SH.deep}.`,
      text: 'Step 2: the frozen text encoder turns each prompt into one embedding, with the deep tokens overwriting its first positions in the first 9 layers.',
      image: `Step 3: a batch of auxiliary images, ${SH.img}, from the other benchmark's test split, with labels and masks.`,
      vision: `Step 4: frozen ViT-L/14@336px. Class token from the original attention; patch tokens from the DPAM path (V-V attention from layer 6): ${gsz}${x}${gsz} (our computation).`,
      scores: 'Step 5: softmax over {g_n, g_a} for the class token (image score) and for every patch token (maps S_n, S_a), upsampled to 518.',
      loss: 'Step 6: cross-entropy on the image label + 4 x (focal + Dice + Dice) on the maps against the mask.',
      backward: 'Backward: red arrows run from the loss into the scores, through g_n and g_a, back through the brown text encoder (frozen, but the gradient passes through) into the green tokens. The image encoder is brown and nothing upstream of it trains: the gradient stops.',
      update: 'Update: Adam, learning rate 0.001, batch 8, 15 epochs, one RTX 3090. Only the green boxes change.',
    })[s],
  });
  // Hovering a term of the loss equation opens the loss stage, where the matching words carry the same data-link.
  const fig = document.getElementById('viz-aclipflow');
  fig.addEventListener('pe-link', e => { if (['total', 'global', 'local'].includes(e.detail)) { const b = document.getElementById('btn-aclipflow-loss'); if (b.getAttribute('aria-pressed') !== 'true') b.click(); } });
})();
</script>

<script>
(() => {
  // ch-synthetic: DRAEM v2 Table 3 ablation (image AUROC on MVTec AD, one model per product) as bars, plus PaDiM's
  // image AUROC as printed in DRAEM's Table 1 for reference. Values are copied from notes/claims/flows-synthetic.json.
  const fig = document.getElementById('viz-draemabl'), el = document.getElementById('cv-draemabl'), out = fig.querySelector('.readout');
  const ROWS = [
    { k: 'reconae', label: 'Reconstruction only (SSIM map)', v: 83.9, g: 'design' },
    { k: 'bozic', label: 'Bozic et al. segmenter, same fakes', v: 92.8, g: 'design' },
    { k: 'disc', label: 'Discriminative subnet only', v: 93.9, g: 'design' },
    { k: 'color', label: 'Fakes: flat random colors', v: 96.2, g: 'realism' },
    { k: 'rect', label: 'Fakes: rectangles, not Perlin', v: 96.9, g: 'realism' },
    { k: 'noaug', label: 'Fakes: no aug., no random opacity', v: 97.4, g: 'realism' },
    { k: 'full', label: 'Full DRAEM', v: 98.0, g: 'full' },
  ];
  const PADIM = 95.5, DEF = { focus: 'all' }, S = { ...DEF, u: 1, prev: 'all' };
  const view = PE.canvas(el, w => w < 560 ? .95 : .42, (ctx, W, H) => draw(ctx, W, H));
  const lit = (r, f) => f === 'all' || r.g === f || r.g === 'full';
  function draw(ctx, W, H) {
    if (!W) return;
    const narrow = W < 560, C = { red: PE.css('--red'), blue: PE.css('--blue'), muted: PE.css('--muted'), fg: PE.css('--fg'), rule: PE.css('--rule'), green: PE.css('--green') };
    ctx.clearRect(0, 0, W, H);
    const lx = narrow ? 8 : 230, b = { x: lx + (narrow ? 0 : 8), y: narrow ? 30 : 16, w: W - lx - (narrow ? 50 : 70), h: H - (narrow ? 64 : 46) };
    const A = PE.axes(ctx, b, [80, 100], [0, 1], { xt: [80, 85, 90, 95, 100], xl: 'image AUROC on MVTec AD' });
    const rowH = b.h / ROWS.length;
    ROWS.forEach((r, i) => {
      const y = b.y + i * rowH + rowH * .18, h = rowH * (narrow ? .38 : .64), a = PE.lerp(lit(r, S.prev) ? 1 : .25, lit(r, S.focus) ? 1 : .25, S.u);
      const col = r.g === 'full' ? C.green : r.g === 'design' ? C.red : C.blue;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.fillRect(A.X(80), narrow ? y + 14 : y, A.X(r.v) - A.X(80), h);
      PE.label(ctx, r.v.toFixed(1), A.X(r.v) + 5, (narrow ? y + 14 : y) + h * .5 + 4, C.fg, 12);
      PE.label(ctx, r.label, narrow ? b.x : lx - 6, narrow ? y + 9 : y + h * .5 + 4, C.fg, 12, narrow ? 'left' : 'right');
      ctx.globalAlpha = 1;
    });
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.muted; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(A.X(PADIM), b.y - 4); ctx.lineTo(A.X(PADIM), b.y + b.h); ctx.stroke(); ctx.restore();
    // The first row's bar is short, so the reference label sits there, clear of every bar.
    PE.label(ctx, narrow ? 'PaDiM (DRAEM\'s table)' : 'PaDiM, as printed by DRAEM', A.X(PADIM) + (narrow ? -4 : 4), b.y + rowH * (narrow ? .75 : .5) + 4, C.muted, 11, narrow ? 'right' : 'left');
  }
  function say() {
    out.textContent = {
      all: 'Full DRAEM scores 98.0. Taking the design apart (red) costs the most; making the fakes unrealistic (blue) costs much less.',
      realism: 'Unrealistic fakes: flat random colors score 96.2 and rectangles 96.9, both above the 95.5 printed for PaDiM. Realism helps, but it is not what makes DRAEM work.',
      design: 'Design taken apart: without the reconstruction subnet the same fakes give 93.9; the supervised segmenter of Bozic et al. gives 92.8; reconstruction alone gives 83.9.',
    }[S.focus];
  }
  function setFocus(f) { if (S.focus === f) return; S.prev = S.focus; S.focus = f;
    ['all', 'realism', 'design'].forEach(k => document.getElementById('btn-draemabl-' + k).setAttribute('aria-pressed', k === f));
    PE.animate(500, u => { S.u = u; view.redraw(); }, () => { S.u = 1; S.prev = f; view.redraw(); }); say(); }
  ['all', 'realism', 'design'].forEach(k => document.getElementById('btn-draemabl-' + k).addEventListener('click', () => setFocus(k)));
  document.getElementById('btn-draemabl-reset').addEventListener('click', () => setFocus(DEF.focus));
  say();
})();
</script>

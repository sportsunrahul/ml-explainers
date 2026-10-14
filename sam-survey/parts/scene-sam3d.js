/* ch-sam3d: from a mask to an object (toy). Tabs: 0 the image and the mask; 1 the coarse shape as voxels in the
   object's own frame (SAM 3D: 64^3, drawn here at 12^3); 2 the layout: rotation (two columns of a rotation matrix,
   the 6D form), translation and an anisotropic scale place the shape in camera space; 3 texture and refinement. */
SCENES.sam3d = ctx => {
  const el = mkCanvas(ctx.stage, 'pe3');
  const S = { step: 0, u: 1 };
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 38, dist: ctx.stage.clientWidth < 620 ? 9.5 : 5.6, yaw: .55, pitch: .3 });
  // canonical voxels of a mug (12^3 grid in [-.5,.5]^3): wall shell, bottom, handle ring
  const n = 12, vox = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    const x = (i + .5) / n - .5, y = (j + .5) / n - .5, z = (k + .5) / n - .5, r = Math.hypot(x, z);
    const body = r < .3 && r > .2 && y > -.42 && y < .38 || r < .3 && y > -.42 && y < -.32;
    const hr = Math.hypot(x - .34, y + .02), handle = Math.abs(z) < .06 && hr > .09 && hr < .17 && x > .3;
    if (body || handle) vox.push({ p: [x, y, z], logo: Math.abs(Math.atan2(z, x) + Math.PI / 2) < .45 && Math.abs(y) < .12 && r > .22 }); }
  const PLACE = { yaw: .7, s: [1.05, 1.25, 1.05], t: [.9, -.35, .6] };
  const rot = (p, a) => [p[0] * Math.cos(a) + p[2] * Math.sin(a), p[1], -p[0] * Math.sin(a) + p[2] * Math.cos(a)];
  const CAN = [-1.3, .35, .9], CANS = [1.4, 1.4, 1.4];
  function placeP(p, u) { // canonical (object frame, left) -> placed (camera frame, on the table), u in 0..1
    const a = rot(p.map((v, i) => v * PE.lerp(CANS[i], PLACE.s[i], u)), PLACE.yaw * u); return a.map((v, i) => v + PE.lerp(CAN[i], PLACE.t[i], u)); }
  // the image plane at the back (z = -1.6), with the mask region of the mug
  function imagePlane(maskA) { const items = [], W = 3.2, H = 2, z = -1.6, nx = 32, ny = 20;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const x0 = -W / 2 + W * i / nx, x1 = x0 + W / nx, y1 = H / 2 - H * j / ny, y0 = y1 - H / ny;
      const wx = (i + .5) / nx * TOY.WW, wy = (j + .5) / ny * TOY.WH, inMug = TOY.inside(TOY.byId.mug, wx, wy), table = wy > .62;
      items.push({ type: 'quad', pts: [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], fill: inMug && maskA > 0 ? PE3.mix(PE.css(table ? '--panel2' : '--panel'), PE.css('--green'), .55 * maskA) : (table ? '--panel2' : '--panel'), stroke: inMug && maskA > 0 ? undefined : '--rule', link: inMug ? 'mask' : 'image' }); }
    items.push({ type: 'label', at: [-W / 2, H / 2 + .12, z], text: 'image I', color: '--blue', align: 'left' });
    if (maskA > 0) items.push({ type: 'label', at: [.75, .5, z], text: 'mask M', color: '--green', alpha: maskA });
    return items; }
  function draw() {
    const u = S.u, items = [{ type: 'grid', y: -1, size: 3.4, step: .4, color: '--rule' }];
    items.push(...imagePlane(S.step >= 0 ? 1 : 0));
    if (S.step >= 1) {
      const lay = S.step >= 2 ? (S.step === 2 ? u : 1) : 0, appear = S.step === 1 ? u : 1, tex = S.step === 3 ? u : 0;
      const pts = vox.map(v => placeP(v.p, lay)), cols = vox.map(v => tex > 0 ? (v.logo ? PE3.mix(PE.css('--green'), PE.css('--fg'), .1 + .4 * (1 - tex)) : PE3.mix(PE.css('--green'), PE.css('--fg'), .55 * tex + .1)) : PE.css('--green'));
      items.push({ type: 'points', pts, colors: cols, size: 3.6 + 1.4 * tex, alpha: appear, link: 'shape' });
      if (lay < .02) items.push({ type: 'label', at: [CAN[0], CAN[1] + .95, CAN[2]], text: 'shape O in its own frame', color: '--green', alpha: appear });
      if (S.step >= 2) { const o = placeP([0, 0, 0], lay), cx = placeP([1.1, 0, 0], lay), cz = placeP([0, 0, 1.1], lay), tt = Math.min(1, lay * 1.4);
        items.push({ type: 'line', pts: [[0, -1, 2.1], o], color: '--yellow', width: 2, dash: [5, 4], alpha: tt, link: 'layout' },
          { type: 'label', at: [0, -1, 2.1], text: 'camera', color: '--muted' },
          { type: 'line', pts: [o, cx], color: '--blue', width: 3, arrow: true, alpha: tt, link: 'layout' }, { type: 'line', pts: [o, cz], color: '--blue', width: 3, arrow: true, alpha: tt, link: 'layout' },
          { type: 'label', at: cx, text: 'R: two columns (6D)', color: '--blue', dx: 6, alpha: tt }, { type: 'label', at: [PE.lerp(0, o[0], .5), -1, PE.lerp(2.1, o[2], .5)], text: 't: translation', color: '--yellow', alpha: tt, dy: 14 },
          { type: 'label', at: placeP([0, .62, 0], lay), text: 's: scale per axis', color: '--muted', alpha: tt }); }
      if (S.step === 3) items.push({ type: 'label', at: placeP([0, -.75, 0], 1), text: 'texture and refined shape', color: '--fg', alpha: tex });
    }
    s3.items = items; s3.redraw();
  }
  const text = ['The input is one image and a mask of one object. SAM 3D does not make the mask itself; it can come from SAM or any segmenter.',
    'The geometry model generates a coarse shape: a 64 x 64 x 64 voxel grid in the object\'s own frame (drawn here at 12 x 12 x 12), even for the hidden back.',
    'The same model predicts the layout: a rotation in 6D form (two columns of a rotation matrix), a translation and a scale per axis, which place the shape in camera space.',
    'A second model refines the shape and adds texture; decoders turn it into a mesh or Gaussian splats.'];
  function readout() { ctx.hud(badge('toy', text[S.step])); }
  let cancel = () => {};
  function animateIn() { cancel(); cancel = s3.tween(900, u => { S.u = u; draw(); }); }
  readout(); draw();
  return { step(n) { S.step = n; readout(); animateIn(); }, resize: () => { s3.setView({ dist: ctx.stage.clientWidth < 620 ? 9.5 : 5.6 }, 0); draw(); }, destroy: () => { cancel(); s3.destroy(); } };
};

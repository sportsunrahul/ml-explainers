// Position ids for [text, image, text, (video, text)] under 1D RoPE and M-RoPE, the rule viz-mrope.js uses.
function build(video) {
  const seq = [];
  const text = ws => ws.forEach(w => seq.push({ kind: 'text', w }));
  const grid = (kind, F, R, C) => { for (let f = 0; f < F; f++) for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) seq.push({ kind, f, r, c, F, R, C }); };
  text(['Read', 'this', 'receipt:']); grid('image', 1, 3, 4); text(['The', 'total', 'is', '11.25.']);
  if (video) { text(['Now', 'this', 'clip:']); grid('video', 2, 2, 3); text(['What', 'moved?']); }
  // M-RoPE: text (n,n,n); visual block starting at s: (s+f, s+r, s+c); next start = max over all components + 1.
  let n = 0;
  for (let i = 0; i < seq.length;) {
    const t = seq[i];
    if (t.kind === 'text') { t.m = [n, n, n]; n++; i++; continue; }
    const s = n; let mx = s;
    while (i < seq.length && seq[i].kind === t.kind && (i === 0 || seq[i - 1].kind === t.kind || seq[i] === t)) {
      const u = seq[i]; u.m = [s + u.f, s + u.r, s + u.c]; mx = Math.max(mx, ...u.m); i++;
      if (i < seq.length && seq[i].kind !== t.kind) break;
    }
    n = mx + 1;
  }
  seq.forEach((t, i) => t.one = i);
  return seq;
}
for (const v of [false, true]) {
  const s = build(v);
  console.log('video', v, 'tokens', s.length, 'max 1D', s.length - 1, 'max M-RoPE', Math.max(...s.map(t => Math.max(...t.m))));
  console.log(s.map(t => (t.w || t.kind[0] + t.f + t.r + t.c) + ':' + t.one + '/(' + t.m.join(',') + ')').join('  '));
}

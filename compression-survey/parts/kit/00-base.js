/* Figure kits shared by explainer.html and atlas.html. Each kit draws into a canvas from a state object and knows
   nothing about buttons: the long page and the atlas each bind their own controls. ASCII only (atlas build checks). */
const KIT = {};
KIT.fmt = (x, d = 2) => {
  const a = Math.abs(x);
  if (a >= 1e12) return (x / 1e12).toFixed(d) + 'T';
  if (a >= 1e9) return (x / 1e9).toFixed(d) + 'G';
  if (a >= 1e6) return (x / 1e6).toFixed(d) + 'M';
  if (a >= 1e3) return (x / 1e3).toFixed(d) + 'k';
  return (+x.toFixed(d)).toString();
};
KIT.bytes = b => b >= 1048576 ? (b / 1048576).toFixed(2) + ' MiB' : b >= 1024 ? (b / 1024).toFixed(1) + ' KiB' : Math.round(b) + ' B';
KIT.bits = (q, n = 8, signed = false) => {
  const v = signed && q < 0 ? q + (1 << n) : q;
  const s = (v >>> 0).toString(2).padStart(n, '0').slice(-n);
  return n > 4 ? s.slice(0, n - 4) + ' ' + s.slice(n - 4) : s;
};
KIT.times = '\u00d7';
// Tween a numeric state object toward a target with PE.animate; returns a cancel function.
KIT.tween = (obj, to, ms, redraw, done) => {
  const from = {}; for (const k in to) from[k] = obj[k];
  return PE.animate(ms, u => { for (const k in to) obj[k] = from[k] + (to[k] - from[k]) * u; redraw(); }, done);
};
KIT.narrow = w => w < 560;

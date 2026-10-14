/* CoreHero: the page's argument as one 3D motion, shared by the long page hero and atlas stage 1.
   Tokens (blue) arrive along a time axis. Each one adds a column of key and value cubes to attention's KV cache
   (brown), which grows; the SSM state (yellow) is a fixed block whose cells only change shade.
   CoreHero.items(u) returns pe3 items for progress u in [0, 1]; u = 1 is the static final diagram. */
const CoreHero = (() => {
  const T = 20, LAY = 4, NST = 4, S = .135, G = .17;   // tokens, layers, state cells per layer, cube size, pitch
  // Centered on the origin (pe3 looks at it): cache block, a gap, then the state block.
  const X0 = -2.32, XS = X0 + (T - 1) * G + .9, Y0 = -.08, TZ = 1.0, TY = -.33;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sm = (a, b, u) => clamp((u - a) / (b - a), 0, 1);
  function cube(c, s, fill, stroke, link, alpha) {
    const h = s / 2, [x, y, z] = c, P = (i, j, k) => [x + i * h, y + j * h, z + k * h];
    const faces = [
      [P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)], [P(-1, -1, -1), P(-1, 1, -1), P(1, 1, -1), P(1, -1, -1)],
      [P(-1, 1, -1), P(-1, 1, 1), P(1, 1, 1), P(1, 1, -1)], [P(-1, -1, -1), P(1, -1, -1), P(1, -1, 1), P(-1, -1, 1)],
      [P(1, -1, -1), P(1, 1, -1), P(1, 1, 1), P(1, -1, 1)], [P(-1, -1, -1), P(-1, -1, 1), P(-1, 1, 1), P(-1, 1, -1)]];
    return faces.map(pts => ({ type: 'quad', pts, fill, stroke, link, alpha }));
  }
  // Deterministic per-(token, layer, cell) shade so the state visibly changes with each token.
  const shade = (t, l, k) => .35 + .65 * ((Math.sin(12.9898 * (t + 1) + 78.233 * (l + 1) + 37.719 * (k + 1)) * 43758.5453) % 1 + 1) % 1;
  // Colors are resolved here (mixes need real values), so callers rebuild items on a theme change.
  function items(u, narrow = false) {
    const mix = PE3.mix, css = PE.css;
    const out = [{ type: 'grid', y: TY - .06, size: 3.6, step: .3, color: '--rule' }];
    // Token clock: token t arrives during [t / T, (t + 1) / T] of the first 85% of u.
    const k = clamp(u / .85, 0, 1) * T, done = Math.floor(k), part = k - done;
    // Time axis and tokens.
    out.push({ type: 'line', pts: [[X0 - .25, TY, TZ], [XS + (NST - 1) * G + .45, TY, TZ]], color: '--muted', width: 1.2, arrow: true, alpha: .7 });
    const tokX = t => X0 + t * G;
    const toks = [];
    for (let t = 0; t < Math.min(T, done + 1); t++) {
      const p = t < done ? 1 : sm(0, .45, part);
      toks.push([tokX(t), TY + .08, TZ + (1 - p) * .9]);
    }
    if (toks.length) out.push({ type: 'points', pts: toks, color: '--blue', size: narrow ? 4.5 : 7, link: 'tokens', alpha: .95 });
    // KV cache: one column of K and V cubes per finished token, per layer.
    const kvB = css('--brown');
    for (let t = 0; t < Math.min(T, done + 1); t++) {
      const g = t < done ? 1 : sm(.35, 1, part);
      if (g <= 0) continue;
      for (let l = 0; l < LAY; l++) {
        const y = Y0 + l * G;
        out.push(...cube([tokX(t), y, -.12], S * g, kvB, kvB, 'cache', .95));
        out.push(...cube([tokX(t), y, .1], S * g, mix(kvB, css('--bg'), .38), kvB, 'cache', .95));
      }
    }
    // SSM state: fixed NST x LAY block; the newest token recolors it.
    const yel = css('--yellow'), bg = css('--bg');
    const tNow = Math.min(T - 1, done), flash = done < T ? Math.sin(Math.PI * sm(.35, 1, part)) : 0;
    // The arriving token is written into the state: a blue stroke from the token to the block while it flashes.
    if (flash > .02) out.push({ type: 'line', pts: [[tokX(tNow), TY + .08, TZ], [XS + (NST - 1) * G / 2, Y0 - G / 2, 0]], color: '--blue', width: 2, alpha: .8 * flash, link: 'tokens' });
    for (let l = 0; l < LAY; l++) for (let c = 0; c < NST; c++) {
      const a = done === 0 && part < .35 ? .25 : shade(tNow, l, c);
      const f = mix(bg, yel, clamp(a + .5 * flash, .18, 1));
      out.push(...cube([XS + c * G, Y0 + l * G, 0], S, f, yel, 'state', 1));
    }
    // Labels settle in at the end.
    const lab = sm(.86, 1, u);
    const nKV = Math.min(T, done + (part > .35 ? 1 : 0));
    const top = Y0 + LAY * G + .12;
    out.push({ type: 'label', at: [X0 + (T - 1) * G / 2, top, 0], text: narrow ? `KV cache: ${nKV} columns` : `KV cache: ${nKV} of ${T} token columns, and growing`, color: '--brown', dy: -10, alpha: Math.max(.55, lab) });
    out.push({ type: 'label', at: [XS + (NST - 1) * G / 2, top, 0], text: narrow ? 'state: fixed' : 'SSM state: same size forever', color: '--yellow', dy: -10, alpha: Math.max(.55, lab) });
    out.push({ type: 'label', at: [XS + (NST - 1) * G + .3, TY, TZ], text: 'time', color: '--muted', dy: 16, alpha: lab });
    if (lab > 0) out.push({ type: 'label', at: [X0 + (T - 1) * G / 2, TY, TZ], text: 'tokens', color: '--blue', dy: 20, alpha: lab });
    return out;
  }
  // Camera that fits the scene's width: pe3's field of view is vertical, so a squarer or portrait stage needs the
  // camera farther back, and a narrow one turns the long cache block toward the viewer.
  function view(W, H) {
    const narrow = W < 620 || W / H < 1.05, r = Math.max(.3, W / Math.max(H, 1));
    return { narrow, yaw: narrow ? .78 : .38, pitch: .34, dist: Math.max(6.2, (narrow ? 5.8 : 5.6) / (.611 * r)) };
  }
  return { items, view, T, LAY, NST };
})();

/* CoreRecurrence: one cell of a recurrence, h_t = lambda h_(t-1) + b x_t with lambda = r e^(i theta), b = 1, against the
   same cell passed through tanh. Exact arithmetic. Shared by the long page (viz-recurrence) and atlas stage 2.
   State S = { r, th, m }: m = 0 is the linear cell, m = 1 the tanh cell, values between are a crossfade for the tween. */
const CoreRecurrence = (() => {
  const K = 60, RMAX = 1.1, THMAX = Math.PI, YMAX = 1.15;
  const PRESETS = { reset: { r: .9, th: 0, m: 0 }, fast: { r: .6, th: 0 }, long: { r: .99, th: 0 }, ring: { r: .9, th: .6 }, explode: { r: 1.1, th: 0 } };

  // Impulse response of the linear cell: one token of size 1 at k = 0, so h_k = lambda^k.
  function linear(r, th) {
    const re = [], mag = [];
    for (let k = 0; k <= K; k++) { const a = Math.pow(r, k); mag.push(a); re.push(a * Math.cos(k * th)); }
    return { re, mag };
  }
  // The tanh cell is the real 2-unit form of lambda (a rotation by th scaled by r), squashed elementwise:
  // h_k = tanh(A h_(k-1) + b x_k). Without tanh it equals the complex cell exactly. grad is |d h_k / d x_0|, carried forward.
  function tanhCell(r, th) {
    const c = Math.cos(th), s = Math.sin(th);
    let h = [Math.tanh(1), 0], J = [1 - h[0] * h[0], 0];
    const re = [h[0]], mag = [Math.hypot(h[0], h[1])], grad = [Math.hypot(J[0], J[1])];
    for (let k = 1; k <= K; k++) {
      const p = [r * (c * h[0] - s * h[1]), r * (s * h[0] + c * h[1])], q = [r * (c * J[0] - s * J[1]), r * (s * J[0] + c * J[1])];
      h = [Math.tanh(p[0]), Math.tanh(p[1])]; J = [q[0] * (1 - h[0] * h[0]), q[1] * (1 - h[1] * h[1])];
      re.push(h[0]); mag.push(Math.hypot(h[0], h[1])); grad.push(Math.hypot(J[0], J[1]));
    }
    return { re, mag, grad };
  }
  const firstBelow = (a, f) => { for (let k = 0; k < a.length; k++) if (a[k] < f * a[0]) return k; return null; };
  // Step at which a linear cell falls below 1% of the token (null if it never does), and reaches 100x (null if it never does).
  const below1 = r => r <= 0 ? 1 : r >= 1 ? null : Math.ceil(Math.log(.01) / Math.log(r));
  const grow100 = r => r <= 1 ? null : Math.ceil(Math.log(100) / Math.log(r));
  // LRU Eq. 6: variance of one cell under uncorrelated unit inputs tends to 1/(1 - |lambda|^2); no limit at |lambda| >= 1.
  const gain = r => r < 1 ? 1 / (1 - r * r) : Infinity;

  const fr = r => Math.abs(r * 100 - Math.round(r * 100)) > 1e-6 ? r.toFixed(3) : r.toFixed(2);
  const pct = v => v >= .1 ? `${(100 * v).toFixed(1)}%` : v >= 1e-4 ? `${(100 * v).toFixed(2)}%` : 'under 0.01%';
  const fg = g => g >= 100 ? Math.round(g).toLocaleString('en-US') : g.toFixed(1);

  function readout(S) {
    const r = S.r, th = S.th, head = `r = ${fr(r)}, \u03b8 = ${th.toFixed(2)} rad`;
    const turn = th > .005 ? ` The cell turns ${th.toFixed(2)} rad per step (a full turn every ${(2 * Math.PI / th).toFixed(1)} steps), so its real part rings.` : '';
    if (S.m >= .5) {
      const T = tanhCell(r, th), sb = firstBelow(T.mag, .01), gb = firstBelow(T.grad, .01);
      const state = sb !== null ? `What is left of the token falls below 1% after ${sb} steps.`
        : r > 1 ? `The state cannot explode: tanh holds it near ${T.mag[K].toFixed(2)}.` : `What is left of the token stays above 1% for all ${K} steps (${pct(T.mag[K] / T.mag[0])} at step ${K}).`;
      const g = gb !== null ? `the toy gradient back to the token falls below 1% after ${gb} steps.` : `the toy gradient back to the token stays above 1% for ${K} steps.`;
      return `tanh RNN, ${head}. The state is no longer a fixed weighted sum of the tokens, so step t must wait for step t-1. ${state} Meanwhile ${g}${turn}`;
    }
    const b = below1(r), up = grow100(r), m50 = Math.pow(r, 50), G = gain(r);
    let fate;
    if (r > 1) fate = `A token's share grows as ${fr(r)}^k: 100 times its size after ${up} steps. The noise gain has no finite value; the state explodes.`;
    else if (r === 1) fate = 'A token never fades: |\u03bb|^k = 1 forever. The noise gain has no finite value; under steady input the state keeps growing.';
    else fate = `A token's share shrinks as ${fr(r)}^k: below 1% after ${b} steps${b > K ? ' (beyond this plot)' : ''}; 50 steps back it keeps ${pct(m50)}. ` +
      `Under random unit input the state's variance is ${fg(G)} times the input's (LRU Eq. 6).`;
    return `Linear cell, ${head}. ${fate} Each past token has a fixed weight \u03bb^(t-k), so all the states can be computed at once.${turn}`;
  }

  function layout(W, H) {
    const narrow = W < 520;
    if (narrow) {
      const x0 = 30, xr = W - 82, ys = 34, hs = H * .37, hb = Math.max(18, (hs - 64) / 2);
      const cy = ys + hs + 30;
      return { narrow, n: 10, x0, xr, yT: ys + 22, yB: ys + 40 + hb, hb, reg: [W - 40, ys + 40 + hb], cs: 13,
        chart: { x: 40, y: cy, w: W - 52, h: Math.max(80, H - 74 - cy) }, gainH: { x: 40, y: H - 22, w: W - 52 } };
    }
    const x0 = 56, xr = W - 196, ys = 30, hs = H * .4, hb = Math.max(24, (hs - 70) / 2);
    const cy = ys + hs + 38;
    return { narrow, n: 16, x0, xr, yT: ys + 26, yB: ys + 46 + hb, hb, reg: [W - 92, ys + 46 + hb], cs: 18,
      chart: { x: 56, y: cy, w: W - 252, h: Math.max(100, H - 44 - cy) }, gainV: { x: W - 118, y: cy + 26, h: Math.max(60, H - 44 - cy - 26) } };
  }

  // The state register: the page's yellow grid of cells. Column 0 is the channel this figure follows (its real and
  // imaginary parts); the other columns stand for the rest of a diagonal state, each cell with its own lambda.
  function register(ctx, cx, cy, cs, C, g, narrow) {
    const cols = narrow ? 2 : 4, gap = 3, w = cols * cs + (cols - 1) * gap, h = 2 * cs + gap, x = cx - w / 2, y = cy - h / 2;
    for (let i = 0; i < cols; i++) for (let j = 0; j < 2; j++) {
      ctx.save(); ctx.globalAlpha = i === 0 ? .9 : .28; ctx.fillStyle = C.yel; ctx.fillRect(x + i * (cs + gap), y + j * (cs + gap), cs, cs); ctx.restore();
      ctx.save(); ctx.strokeStyle = C.yel; ctx.lineWidth = i === 0 ? 1.5 + 2.5 * g : 1; ctx.strokeRect(x + i * (cs + gap), y + j * (cs + gap), cs, cs); ctx.restore();
    }
    PE.label(ctx, 'state h_t', cx, narrow ? y + h + 14 : y - 8, C.yel, 12, 'center');
    if (!narrow) PE.label(ctx, 'fixed size', cx, y + h + 16, C.mu, 11, 'center');
    return { x, y, w, h };
  }

  function draw(ctx, W, H, S, glow = () => 0) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), blue: PE.css('--blue'), yel: PE.css('--yellow'), red: PE.css('--red') };
    ctx.clearRect(0, 0, W, H);
    const Lo = layout(W, H), m = PE.clamp(S.m, 0, 1), lin = 1 - m;
    const Li = linear(S.r, S.th), T = m > 0 ? tanhCell(S.r, S.th) : null;
    const gT = glow('token'), gB = glow('bars'), gS = glow('state'), gM = glow('magnitude'), gP = glow('phase'), gG = glow('gain');

    // Strip: the last n tokens, oldest on the left, and the state register at "now".
    const n = Lo.n, X = i => Lo.x0 + i * (Lo.xr - Lo.x0) / (n - 1), R = register(ctx, Lo.reg[0], Lo.reg[1], Lo.cs, C, gS, Lo.narrow);
    PE.polyline(ctx, [[Lo.x0 - 14, Lo.yT], [Lo.xr + 18, Lo.yT]], C.mu, 1, null, .6);
    PE.arrow(ctx, Lo.xr + 18, Lo.yT, Lo.xr + 30, Lo.yT, C.mu, 1, 7, .6);
    PE.label(ctx, Lo.narrow ? 'time' : 'time, oldest token on the left', Lo.x0 - 14, Lo.yT - 12, C.mu, 11);
    // Linear: one bar per token, Re lambda^(t-k), and a line from each into the state.
    if (lin > .01) {
      PE.polyline(ctx, [[Lo.x0 - 10, Lo.yB], [Lo.xr + 10, Lo.yB]], C.rule, 1, null, lin);
      const bw = Math.max(3, (Lo.xr - Lo.x0) / (n - 1) * .42) * (1 + .5 * gB);
      for (let i = 0; i < n; i++) {
        const lag = n - 1 - i, mg = Math.pow(S.r, lag), v = mg * Math.cos(lag * S.th), hgt = PE.clamp(v, -1, 1) * Lo.hb;
        PE.polyline(ctx, [[X(i), Lo.yB - Math.max(0, hgt)], [R.x - 4, Lo.reg[1]]], C.blue, 1, null, lin * (.08 + .35 * Math.min(1, mg)));
        ctx.save(); ctx.globalAlpha = lin * (.75 + .25 * gB); ctx.fillStyle = C.blue;
        ctx.fillRect(X(i) - bw / 2, Math.min(Lo.yB, Lo.yB - hgt), bw, Math.max(1, Math.abs(hgt))); ctx.restore();
        if (mg > 1) PE.arrow(ctx, X(i), Lo.yB - Lo.hb + 18, X(i), Lo.yB - Lo.hb + 5, C.red, 1.5, 6, lin);
      }
      PE.label(ctx, Lo.narrow ? 'bars: weight \u03bb^(t-k)' : 'linear: each token sits in the state with a fixed weight \u03bb^(t-k) (bars)', Lo.x0 - 10, Lo.yB + Lo.hb + 20, C.blue, 11, 'left');
    }
    // tanh: no weights; a chain of states, each waiting for the one before.
    if (m > .01) {
      const ny = Lo.yB, cs = Lo.narrow ? 5 : 6;
      for (let i = 0; i < n; i++) {
        PE.arrow(ctx, X(i), Lo.yT + 20, X(i), ny - cs - 3, C.blue, 1.2, 5, m * .8);
        ctx.save(); ctx.globalAlpha = m * .9; ctx.fillStyle = C.yel; ctx.fillRect(X(i) - cs / 2, ny - cs, cs, cs); ctx.fillRect(X(i) - cs / 2, ny + 1, cs, cs); ctx.restore();
        const xe = i < n - 1 ? X(i + 1) - cs : R.x - 4, ye = i < n - 1 ? ny : Lo.reg[1];
        PE.arrow(ctx, X(i) + cs, ny, xe, ye, C.yel, 1.2, 5, m * .7);
      }
      const tx = (X(n - 2) + X(n - 1)) / 2;
      PE.label(ctx, 'tanh', tx, ny - 10, C.yel, 10, 'center');
      if (!Lo.narrow) PE.label(ctx, 'tanh', (X(0) + X(1)) / 2, ny - 10, C.yel, 10, 'center');
      PE.label(ctx, Lo.narrow ? 'tanh: no fixed weights; one step at a time' : 'tanh: no fixed weight per token; each state must wait for the one before it', Lo.x0 - 10, Lo.yB + Lo.hb + 20, C.yel, 11, 'left');
    }
    for (let i = 0; i < n; i++) PE.dot(ctx, X(i), Lo.yT, (Lo.narrow ? 3.5 : 4.5) * (1 + .5 * gT), C.blue);
    const tl = Lo.narrow ? [0, n - 1] : [0, 5, 10, n - 1];
    tl.forEach(i => PE.label(ctx, i === n - 1 ? 't' : `t-${n - 1 - i}`, X(i), Lo.yT + 15, C.mu, 10, 'center'));

    // Chart: what is left of one token, k steps after it arrived.
    const b = Lo.chart, yt = Lo.narrow ? [-1, 0, 1] : [-1, -.5, 0, .5, 1];
    const ax = PE.axes(ctx, b, [0, K], [-YMAX, YMAX], { xt: [0, 10, 20, 30, 40, 50, 60], yt, yf: v => v === 0 ? '0' : v.toFixed(1).replace('.0', ''), xl: 'steps k since the token arrived' });
    const XC = ax.X, YC = v => ax.Y(PE.clamp(v, -YMAX, YMAX));
    PE.polyline(ctx, [[b.x, YC(0)], [b.x + b.w, YC(0)]], C.mu, 1, null, .6);
    PE.label(ctx, Lo.narrow ? 'one token, k steps later' : 'one token, k steps later: what is left of it in the state (impulse response)', b.x, b.y - 10, C.fg, 12);
    const re = Li.re.map((v, k) => lin * v + (T ? m * T.re[k] : 0)), mag = Li.mag.map((v, k) => lin * v + (T ? m * T.mag[k] : 0));
    // Envelope +-|h_k| (dashed) and the real part as stems: it rings when theta is not 0.
    const env = s => mag.map((v, k) => [XC(k), YC(s * v)]);
    PE.polyline(ctx, env(1), C.yel, 1.6 + 2.5 * gM, [5, 4], .9);
    PE.polyline(ctx, env(-1), C.yel, 1.2 + 2 * gM, [5, 4], .45);
    for (let k = 0; k <= K; k++) {
      PE.polyline(ctx, [[XC(k), YC(0)], [XC(k), YC(re[k])]], C.yel, (Lo.narrow ? 1.2 : 1.8) + 2 * gP, null, .85);
      if (Math.abs(re[k]) <= YMAX) PE.dot(ctx, XC(k), YC(re[k]), (Lo.narrow ? 1.8 : 2.4) + 1.5 * gP, C.yel);
    }
    if (T) PE.polyline(ctx, T.grad.map((v, k) => [XC(k), YC(v)]), C.red, 2.2, null, m);
    const kOut = mag.findIndex(v => v > YMAX);
    if (kOut > 0) {
      PE.arrow(ctx, XC(kOut), b.y + 22, XC(kOut), b.y + 4, C.red, 2, 8);
      const up = grow100(S.r);
      PE.label(ctx, up && lin > .5 ? `off the chart: 100x by step ${up}` : 'off the chart', b.x + b.w, b.y - 10, C.red, 11, 'right');
    }
    // Markers: 50 steps back (the prediction in the text), and the step where the token drops below 1%.
    const k50 = mag[50] / mag[0];
    ctx.save(); ctx.strokeStyle = C.fg; ctx.globalAlpha = .35; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(XC(50), b.y); ctx.lineTo(XC(50), b.y + b.h); ctx.stroke(); ctx.restore();
    if (kOut < 0 || kOut > 50) PE.label(ctx, `k = 50: ${pct(k50)}`, XC(50) - 4, b.y + 12, C.fg, 11, 'right');
    const kb = firstBelow(mag, .01);
    if (kb !== null && kb <= K && kb !== 50) {
      PE.polyline(ctx, [[XC(kb), YC(0) - 8], [XC(kb), YC(0) + 8]], C.fg, 2);
      PE.label(ctx, `below 1% at ${kb}`, PE.clamp(XC(kb), b.x + 40, b.x + b.w - 40), b.y + b.h - 6, C.fg, 11, 'center');
    }
    if (m > .05) PE.label(ctx, Lo.narrow ? 'red: toy gradient' : 'red: toy gradient back to the token, |d h_k / d x_0|', b.x + b.w - 6, b.y + 30, C.red, 11, 'right');

    // Gain 1/(1 - |lambda|^2) on a log scale from 1 to 1000 (LRU Eq. 6).
    // The gain belongs to the linear cell, so it fades out in tanh mode.
    const G = gain(S.r), f = Number.isFinite(G) ? PE.clamp(Math.log10(G) / 3, 0, 1) : 1;
    const gtxt = Number.isFinite(G) ? `${fg(G)}x` : 'no limit';
    ctx.save(); ctx.globalAlpha = .3 + .7 * lin;
    if (Lo.gainV) {
      const g = Lo.gainV, bw = 16 + 6 * gG, x = g.x + 30;
      PE.label(ctx, 'noise gain', g.x + 38, g.y - 26, C.red, 11, 'center');
      PE.label(ctx, '1/(1-|\u03bb|\u00b2)', g.x + 38, g.y - 12, C.mu, 10, 'center');
      ctx.save(); ctx.fillStyle = C.rule; ctx.fillRect(x, g.y, bw, g.h); ctx.globalAlpha *= .85 + .15 * gG; ctx.fillStyle = C.red; ctx.fillRect(x, g.y + g.h * (1 - f), bw, g.h * f); ctx.restore();
      [1, 10, 100, 1000].forEach((v, i) => PE.label(ctx, String(v), x - 5, g.y + g.h * (1 - i / 3) + 4, C.mu, 10, 'right'));
      PE.label(ctx, gtxt, x + bw + 6, PE.clamp(g.y + g.h * (1 - f) + 4, g.y + 10, g.y + g.h), C.red, 12);
      if (m > .5) PE.label(ctx, 'linear cell only', g.x + 38, g.y + g.h + 16, C.mu, 10, 'center');
    } else {
      const g = Lo.gainH;
      ctx.save(); ctx.fillStyle = C.rule; ctx.fillRect(g.x, g.y, g.w, 7 + 3 * gG); ctx.globalAlpha *= .85 + .15 * gG; ctx.fillStyle = C.red; ctx.fillRect(g.x, g.y, g.w * f, 7 + 3 * gG); ctx.restore();
      PE.label(ctx, `noise gain 1/(1-|\u03bb|\u00b2): ${gtxt}${m > .5 ? ', linear cell only' : ' (scale 1 to 1000)'}`, g.x, g.y - 6, C.red, 11);
    }
    ctx.restore();
  }
  return { K, RMAX, THMAX, PRESETS, linear, tanhCell, below1, grow100, gain, readout, draw };
})();

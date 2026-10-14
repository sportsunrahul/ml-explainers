/* Chapter 2, "play it yourself": 5 arms with hidden means (seeded draws from N(0, 1)), rewards ~ N(mean, 1), 20 pulls.
   Each column shows the arm's rewards (yellow dots), its sample-average estimate Q (blue bar) and, once revealed,
   its true mean (brown dashed line: a fixed property of the environment). Epsilon-greedy (epsilon .1) can finish
   the pulls from the reader's estimates. With showUcb, each arm also shows its UCB bound Q + c sqrt(ln t / N), c = 2
   (S&B Eq. 2.10): a pale blue whisker above the estimate, which shrinks as the arm is pulled. */
KIT.BANDIT_SEED = 210;
KIT.bandit = (canvas, o = {}) => {
  const K = 5, BUDGET = 20, EPS = .1;
  const st = {};
  let stop = () => {};
  const glow = o.fig ? PE.linkPulses(o.fig, () => cv.redraw()) : () => 0;
  const reset = () => {
    stop(); const r = PE.rng(KIT.BANDIT_SEED), g = PE.gauss(r);
    Object.assign(st, { means: Array.from({ length: K }, () => g()), Q: new Array(K).fill(0), N: new Array(K).fill(0),
      rewards: Array.from({ length: K }, () => []), total: 0, pulls: 0, last: null, revealed: false, auto: false, flash: 0, showUcb: false });
    st.rand = PE.rng(KIT.BANDIT_SEED * 31 + 1); st.noise = PE.gauss(st.rand);
  };
  const pull = (a, who = 'you') => {
    if (st.pulls >= BUDGET) return null;
    const R = st.means[a] + st.noise(), Qold = st.Q[a];
    st.N[a]++; st.Q[a] += (R - st.Q[a]) / st.N[a];   // incremental sample average (S&B Eq. 2.3)
    st.rewards[a].push(R); st.total += R; st.pulls++; st.last = { a, R, who, Qold };
    stop = PE.animate(500, u => { st.flash = 1 - u; cv.redraw(); });
    return R;
  };
  const egreedyChoice = () => {
    if (st.rand() < EPS) return Math.floor(st.rand() * K);
    const m = Math.max(...st.Q), best = st.Q.map((q, i) => q === m ? i : -1).filter(i => i >= 0);
    return best[Math.floor(st.rand() * best.length)];
  };
  const finish = done => {
    st.auto = true;
    if (PE.instant()) { while (st.pulls < BUDGET) pull(egreedyChoice(), 'eps'); st.flash = 0; cv.redraw(); done && done(); return; }
    const tick = () => { if (st.pulls >= BUDGET) { done && done(); return; } pull(egreedyChoice(), 'eps'); setTimeout(tick, 220); };
    tick();
  };
  const draw = (ctx, w, h) => {
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const FG = PE.css('--fg'), M = PE.css('--muted'), B = PE.css('--blue'), Y = PE.css('--yellow'), BR = PE.css('--brown'), RU = PE.css('--rule'), P2 = PE.css('--panel2');
    const narrow = KIT.narrow(w), top = narrow ? 44 : 34, bot = h - (narrow ? 52 : 48), left = narrow ? 30 : 40, colW = (w - left - 10) / K;
    const lo = -3.2, hi = 3.2, Yv = v => bot - (PE.clamp(v, lo, hi) - lo) / (hi - lo) * (bot - top);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = M; ctx.textAlign = 'right';
    for (const v of [-3, -2, -1, 0, 1, 2, 3]) {
      ctx.strokeStyle = v === 0 ? M : RU; ctx.lineWidth = 1; ctx.globalAlpha = v === 0 ? .6 : 1;
      ctx.beginPath(); ctx.moveTo(left, Yv(v)); ctx.lineTo(w - 10, Yv(v)); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(String(v).replace('-', '\u2212'), left - 6, Yv(v) + 4);
    }
    const gq = glow('estimate'), gr = glow('reward'), ge = glow('error'), gn = glow('count'), gb = glow('bonus'), gc = glow('choice');
    const ucb = KIT.banditUcb(st), pick = ucb.indexOf(Math.max(...ucb));
    for (let a = 0; a < K; a++) {
      const x0 = left + a * colW, cx = x0 + colW / 2;
      ctx.beginPath(); ctx.roundRect(x0 + 4, top - 6, colW - 8, bot - top + 12, 8); ctx.fillStyle = P2; ctx.globalAlpha = .55; ctx.fill(); ctx.globalAlpha = 1;
      const rs = st.rewards[a];
      rs.forEach((R, i) => {
        const isLast = st.last && st.last.a === a && i === rs.length - 1;
        PE.dot(ctx, cx - colW * .22 + ((i * 37) % 11) / 11 * colW * .16, Yv(R), isLast ? 4 + 4 * st.flash : 3.2, Y, isLast ? 1 : .75 + gr * .25);
      });
      if (st.N[a]) {
        ctx.save(); ctx.strokeStyle = B; ctx.lineWidth = 4 + 3 * gq; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(cx - colW * .02, Yv(st.Q[a])); ctx.lineTo(cx + colW * .3, Yv(st.Q[a])); ctx.stroke(); ctx.restore();
      }
      if (st.showUcb || gb) {
        const top = ucb[a] === Infinity ? hi : ucb[a], y1 = Yv(st.N[a] ? st.Q[a] : 0), y2 = Yv(top);
        ctx.save(); ctx.strokeStyle = B; ctx.globalAlpha = .45 + .55 * gb; ctx.lineWidth = 2.5 + 2 * gb;
        ctx.beginPath(); ctx.moveTo(cx + colW * .14, y1); ctx.lineTo(cx + colW * .14, y2); ctx.moveTo(cx + colW * .04, y2); ctx.lineTo(cx + colW * .24, y2); ctx.stroke(); ctx.restore();
        if (ucb[a] === Infinity) PE.label(ctx, 'untried', cx + colW * .14, Yv(hi) - 4, B, 10, 'center');
      }
      if ((st.showUcb || gc) && a === pick && st.pulls < BUDGET) {
        ctx.save(); ctx.strokeStyle = PE.css('--green'); ctx.lineWidth = 2 + 2 * gc; ctx.beginPath(); ctx.roundRect(x0 + 4, top - 6, colW - 8, bot - top + 12, 8); ctx.stroke(); ctx.restore();
      }
      if (ge && st.last && st.last.a === a) {
        ctx.save(); ctx.globalAlpha = ge; PE.polyline(ctx, [[cx + colW * .3, Yv(st.last.Qold)], [cx + colW * .3, Yv(st.last.R)]], PE.css('--red'), 3); ctx.restore();
      }
      if (st.revealed) {
        PE.polyline(ctx, [[x0 + 8, Yv(st.means[a])], [x0 + colW - 8, Yv(st.means[a])]], BR, 2, [5, 4]);
      }
      const best = st.revealed && st.means[a] === Math.max(...st.means);
      PE.label(ctx, 'arm ' + (a + 1), cx, bot + 20, best ? BR : FG, narrow ? 12 : 13, 'center', 700);
      const qs = st.N[a] ? 'Q=' + st.Q[a].toFixed(2).replace('-', '\u2212') : '', ns = st.N[a] ? 'n=' + st.N[a] : 'not pulled';
      if (narrow) { PE.label(ctx, ns, cx, bot + 34, st.N[a] ? B : M, 10, 'center', gn ? 800 : 600); if (qs) PE.label(ctx, qs, cx, bot + 47, B, 10, 'center', 600); }
      else PE.label(ctx, st.N[a] ? ns + '  ' + qs : ns, cx, bot + 38, st.N[a] ? B : M, 12, 'center', gn ? 800 : 600);
    }
    PE.label(ctx, 'pulls ' + st.pulls + ' of ' + BUDGET + '   total reward ' + st.total.toFixed(2).replace('-', '\u2212'), left, narrow ? 14 : 16, Y, narrow ? 12 : 13, 'left', 700);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? .95 : .5, draw);
  reset();
  return { pull: a => { const R = pull(a); cv.redraw(); return R; }, finish, reveal: () => { st.revealed = true; cv.redraw(); },
    showUcb: on => { st.showUcb = on; cv.redraw(); },
    reset: () => { reset(); cv.redraw(); }, state: st, redraw: cv.redraw, resize: cv.resize, destroy: cv.destroy };
};
KIT.banditText = st => {
  const f = x => x.toFixed(2).replace('-', '\u2212');
  if (!st.pulls) return 'Five arms, 20 pulls. Each pull pays a random amount around that arm\'s hidden mean. Try to finish with the largest total.';
  const l = st.last, who = l.who === 'eps' ? 'Epsilon-greedy pulled' : 'You pulled';
  let s = who + ' arm ' + (l.a + 1) + ': it paid ' + f(l.R) + '. Its estimate is now ' + f(st.Q[l.a]) + ' after ' + st.N[l.a] + (st.N[l.a] === 1 ? ' pull.' : ' pulls.');
  if (st.pulls >= 20) {
    const best = st.means.indexOf(Math.max(...st.means));
    s = 'All 20 pulls used: total ' + f(st.total) + '. ' + (st.revealed ? 'The best arm was arm ' + (best + 1) + ' (mean ' + f(st.means[best]) + '); it got ' + st.N[best] + ' of 20 pulls.' : 'Press Reveal to see the true means.');
  }
  return s;
};
// UCB bound per arm after the pulls so far (S&B Eq. 2.10 with c = 2; an unpulled arm's bound is infinite).
KIT.banditUcb = st => st.Q.map((q, a) => st.N[a] ? q + 2 * Math.sqrt(Math.log(Math.max(1, st.pulls)) / st.N[a]) : Infinity);

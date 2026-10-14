<style>
canvas.pe3 { display:block; width:100%; touch-action:pan-y; cursor:grab; }
canvas.pe3.pe3-drag { cursor:grabbing; }
canvas.pe3.pe3-static { cursor:default; }
canvas.pe3:focus-visible { outline:2px solid var(--yellow, #f4d35e); outline-offset:2px; }
</style>
<script>
/* PE3: a small 3D renderer on canvas 2D for point clouds, surfaces, trajectories and stacked planes. Standalone (no PE).
   const s = PE3.scene(canvasEl, {
     aspect: .62,            // height / width; number or function(w)
     fov: 40, dist: 6,       // perspective camera looking at the origin; fov in degrees (vertical)
     yaw: .6, pitch: .35,    // radians; yaw turns around y, pitch > 0 looks down from above
     autoRotate: 0,          // rad/s; forced to 0 under prefers-reduced-motion
     draggable: true,        // pointer drag rotates; arrow keys rotate by .15 rad when the canvas is focused
     depthCue: true,         // farther items a little smaller and fainter
     items: [],              // replace any time: s.items = [...]; s.redraw()
     onFrame(s, t, dt) {} }) // optional; keeps the rAF loop running while the canvas is on screen
   s.project([x,y,z]) -> { x, y, depth, k }   // CSS px; depth = 1 at the origin, > 1 farther away; k = px per world unit there
   s.setView({ yaw, pitch, dist }, ms)        // eased; jumps when PE3.instant()
   s.tween(ms, step(u), done) -> cancel       // eased u in 0..1 (ends on exactly 1); step(1) at once when PE3.instant()
   s.pulse(word)                              // ~1.1 s highlight of items whose link === word; a 'pe-link' CustomEvent on the enclosing <figure> does the same
   s.redraw(); s.destroy()
   PE3.instant()                              // prefers-reduced-motion or <html data-cue-replay="1">
   PE3.surface(f, { x0, x1, z0, z1, nx, nz, fill(y) -> color, stroke }) -> quad items, y = f(x, z)
   PE3.gaussian(n, { mean: [x,y,z], scale: [sx,sy,sz], seed }) -> [[x,y,z], ...]  (mulberry32, deterministic)
   PE3.rng(seed) -> () => uniform [0,1);  PE3.mix(c1, c2, u) -> 'rgb(...)'
   Items (plain objects; all take link and alpha; any color may be '#hex', 'rgb()', a CSS variable name like '--blue', or a function returning one):
     { type:'points', pts, color, colors, size }     size in px at depth 1
     { type:'line',   pts, color, width, dash, arrow }
     { type:'quad',   pts:[p0,p1,p2,p3], fill, stroke }  depth-sorted by centroid; stroke defaults to the fill (hides seams)
     { type:'label',  at, text, color, dx, dy, font, align }  screen-space, never scaled, drawn last
     { type:'axes',   len, color, labels:['x','y','z'] }
     { type:'grid',   y, size, step, color }
   The loop runs only while something moves (view easing, autoRotate, a pulse, a tween, onFrame) and the canvas is on screen. */
const PE3 = (() => {
  const instant = () => matchMedia('(prefers-reduced-motion: reduce)').matches || document.documentElement.dataset.cueReplay === '1';
  const ease = u => u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const PULSE_MS = 1100;

  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gaussian(n, o = {}) {
    const r = rng(o.seed == null ? 1 : o.seed), m = o.mean || [0, 0, 0], sc = o.scale || [1, 1, 1], out = [];
    const g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    for (let i = 0; i < n; i++) out.push([m[0] + sc[0] * g(), m[1] + sc[1] * g(), m[2] + sc[2] * g()]);
    return out;
  }
  function surface(f, o) {
    const nx = o.nx || 24, nz = o.nz || 24, items = [];
    const P = (i, j) => { const x = o.x0 + (o.x1 - o.x0) * i / nx, z = o.z0 + (o.z1 - o.z0) * j / nz; return [x, f(x, z), z]; };
    for (let i = 0; i < nx; i++) for (let j = 0; j < nz; j++) {
      const pts = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)], y = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4;
      items.push({ type: 'quad', pts, fill: typeof o.fill === 'function' ? () => o.fill(y) : o.fill, stroke: o.stroke, link: o.link, alpha: o.alpha });
    }
    return items;
  }

  // Color parsing: '#rgb', '#rrggbb', 'rgb(a)(...)' -> [r,g,b]; anything else is left to the canvas untouched.
  function parse(c) {
    if (!c) return null;
    if (c[0] === '#') { const h = c.length === 4 ? c.slice(1).split('').map(x => x + x).join('') : c.slice(1, 7); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
    const m = /rgba?\(([^)]+)\)/.exec(c); if (m) return m[1].split(/[\s,\/]+/).slice(0, 3).map(Number);
    return null;
  }
  function mix(c1, c2, u) {
    const a = parse(c1), b = parse(c2); if (!a || !b) return u < .5 ? c1 : c2;
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(',')})`;
  }

  function scene(el, o = {}) {
    const ctx = el.getContext('2d');
    const S = {
      items: o.items || [], yaw: o.yaw ?? .6, pitch: o.pitch ?? .35, dist: o.dist ?? 6, fov: o.fov ?? 40,
      autoRotate: o.autoRotate || 0, depthCue: o.depthCue !== false, onFrame: o.onFrame || null, w: 0, h: 0, el,
    };
    const aspect = o.aspect ?? .62;
    const tweens = [], glow = {};   // glow[word] = start time of the pulse
    let viewTween = null, raf = 0, running = false, visible = true, dirty = false, last = 0, dead = false;
    let cssCache = {}, F = 1, cam = null;

    const css = n => cssCache[n] || (cssCache[n] = getComputedStyle(document.documentElement).getPropertyValue(n).trim());
    const col = c => { if (typeof c === 'function') c = c(); return c && c[0] === '-' && c[1] === '-' ? (css(c) || '#888') : c; };
    const fg = () => css('--fg') || '#e8e6e3';

    function camera() {
      const cy = Math.cos(S.yaw), sy = Math.sin(S.yaw), cp = Math.cos(S.pitch), sp = Math.sin(S.pitch);
      F = (S.h / 2) / Math.tan(S.fov * Math.PI / 360);
      cam = { cy, sy, cp, sp };
    }
    function project(p) {
      if (!cam) camera();
      const { cy, sy, cp, sp } = cam;
      const x1 = p[0] * cy - p[2] * sy, z1 = p[0] * sy + p[2] * cy;
      const y2 = p[1] * cp - z1 * sp, z2 = p[1] * sp + z1 * cp;
      const cz = Math.max(.05, S.dist - z2), k = F / cz;
      return { x: S.w / 2 + x1 * k, y: S.h / 2 - y2 * k, depth: cz / S.dist, k };
    }
    const cue = d => S.depthCue ? clamp(1 - (d - 1) * 1.4, .3, 1) : 1;
    const sizeCue = d => S.depthCue ? 1 / d : 1;

    function glowOf(link, now) {
      if (!link || !(link in glow)) return 0;
      return instant() ? 1 : Math.sin(Math.PI * (now - glow[link]) / PULSE_MS);
    }
    const lit = (c, g) => g > 0 ? mix(c, fg(), .55 * g) : c;

    // Expand axes and grid into lines and labels, so the sorter sees one kind of thing.
    function expand(items) {
      const out = [];
      for (const it of items) {
        if (it.type === 'axes') {
          const L = it.len || 1.5, c = it.color || '--muted', lab = it.labels || ['x', 'y', 'z'];
          [[L, 0, 0], [0, L, 0], [0, 0, L]].forEach((e, i) => {
            out.push({ type: 'line', pts: [[0, 0, 0], e], color: c, width: it.width || 1, alpha: it.alpha, link: it.link, arrow: true });
            if (lab[i]) out.push({ type: 'label', at: e.map(v => v * 1.08), text: lab[i], color: c, alpha: it.alpha, link: it.link, font: it.font });
          });
        } else if (it.type === 'grid') {
          const y = it.y || 0, sz = it.size || 2, st = it.step || .5, c = it.color || '--rule', n = Math.round(sz / st);
          for (let i = -n; i <= n; i++) {
            const v = i * st;
            out.push({ type: 'line', pts: [[v, y, -sz], [v, y, sz]], color: c, width: it.width || 1, alpha: it.alpha, link: it.link });
            out.push({ type: 'line', pts: [[-sz, y, v], [sz, y, v]], color: c, width: it.width || 1, alpha: it.alpha, link: it.link });
          }
        } else out.push(it);
      }
      return out;
    }

    function draw(now) {
      cssCache = {}; camera();
      // Expire by time here, not in glowOf: a pulse for a word no item carries must still end, or the loop never idles.
      for (const k in glow) if (now - glow[k] >= PULSE_MS) delete glow[k];
      const W = S.w, H = S.h; if (!W || !H) return;
      ctx.clearRect(0, 0, W, H);
      const draws = [], labels = [];
      for (const it of expand(S.items)) {
        const g = glowOf(it.link, now), a = it.alpha ?? 1;
        if (it.type === 'points') {
          const c = col(it.color || '--blue');
          it.pts.forEach((p, i) => { const q = project(p); draws.push({ d: q.depth, t: 0, q, c: it.colors ? col(it.colors[i]) : c, size: it.size || 3, g, a }); });
        } else if (it.type === 'line') {
          const q = it.pts.map(project); if (!q.length) continue;
          draws.push({ d: q.reduce((s, p) => s + p.depth, 0) / q.length, t: 1, q, it, g, a });
        } else if (it.type === 'quad') {
          const q = it.pts.map(project);
          draws.push({ d: (q[0].depth + q[1].depth + q[2].depth + q[3].depth) / 4, t: 2, q, it, g, a });
        } else if (it.type === 'label') labels.push({ it, q: project(it.at), g, a });
      }
      draws.sort((a, b) => b.d - a.d);
      for (const d of draws) {
        const ga = d.a * cue(d.d);
        if (d.t === 0) {
          const r = d.size * sizeCue(d.d) * (1 + .6 * d.g);
          ctx.globalAlpha = clamp(ga + .4 * d.g, 0, 1); ctx.fillStyle = lit(d.c, d.g);
          ctx.beginPath(); ctx.arc(d.q.x, d.q.y, r, 0, 7); ctx.fill();
        } else if (d.t === 1) {
          const it = d.it, q = d.q;
          ctx.save(); ctx.globalAlpha = clamp(ga + .4 * d.g, 0, 1); ctx.strokeStyle = lit(col(it.color || '--fg'), d.g);
          ctx.lineWidth = (it.width || 1.5) * sizeCue(d.d) * (1 + .6 * d.g); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
          if (it.dash) ctx.setLineDash(it.dash);
          ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke();
          if (it.arrow && q.length > 1) {
            const a = q[q.length - 2], b = q[q.length - 1], an = Math.atan2(b.y - a.y, b.x - a.x), hd = Math.min(9, Math.hypot(b.x - a.x, b.y - a.y) * .5);
            ctx.setLineDash([]); ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.moveTo(b.x, b.y);
            ctx.lineTo(b.x - hd * Math.cos(an - .42), b.y - hd * Math.sin(an - .42)); ctx.lineTo(b.x - hd * Math.cos(an + .42), b.y - hd * Math.sin(an + .42)); ctx.closePath(); ctx.fill();
          }
          ctx.restore();
        } else {
          const it = d.it, q = d.q, f = lit(col(it.fill || '--panel2'), d.g);
          ctx.save(); ctx.globalAlpha = clamp(ga + .3 * d.g, 0, 1);
          ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.closePath();
          if (it.fill !== null) { ctx.fillStyle = f; ctx.fill(); }
          if (it.stroke !== null) { ctx.strokeStyle = it.stroke ? lit(col(it.stroke), d.g) : f; ctx.lineWidth = it.stroke ? (it.width || 1) : .75; ctx.lineJoin = 'round'; ctx.stroke(); }
          ctx.restore();
        }
      }
      ctx.globalAlpha = 1;
      for (const { it, q, g, a } of labels) {
        ctx.save(); ctx.globalAlpha = clamp(a * cue(q.depth) + .5 * g, 0, 1);
        ctx.font = it.font || '600 12px system-ui, sans-serif'; ctx.textAlign = it.align || 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = lit(col(it.color || '--fg'), g); ctx.fillText(it.text, q.x + (it.dx || 0), q.y + (it.dy || 0)); ctx.restore();
      }
    }

    const spin = () => S.autoRotate && !instant() ? S.autoRotate : 0;
    const active = () => viewTween || tweens.length || spin() || Object.keys(glow).length || S.onFrame;
    function frame(t) {
      raf = 0; if (dead) return;
      const dt = last ? (t - last) / 1000 : 0; last = t;
      if (spin()) S.yaw += spin() * dt;
      if (viewTween) {
        const u = clamp((t - viewTween.t0) / viewTween.ms, 0, 1), e = ease(u);
        for (const k in viewTween.to) S[k] = viewTween.from[k] + (viewTween.to[k] - viewTween.from[k]) * e;
        if (u >= 1) { Object.assign(S, viewTween.to); viewTween = null; }
      }
      for (const tw of tweens.slice()) {
        const u = clamp((t - tw.t0) / tw.ms, 0, 1); tw.step(u >= 1 ? 1 : ease(u));
        if (u >= 1) { tweens.splice(tweens.indexOf(tw), 1); tw.done && tw.done(); }
      }
      if (S.onFrame) S.onFrame(S, t, dt);
      draw(t); dirty = false;
      if (active() && visible) raf = requestAnimationFrame(frame); else { running = false; last = 0; }
    }
    function kick() { if (dead || running || !visible) return; running = true; last = 0; raf = requestAnimationFrame(frame); }
    function redraw() { if (running) { dirty = true; return; } if (!visible) { dirty = true; return; } kick(); }

    function tween(ms, step, done) {
      if (instant()) { step(1); done && done(); redraw(); return () => {}; }
      const tw = { t0: performance.now(), ms, step, done }; tweens.push(tw); kick();
      return () => { const i = tweens.indexOf(tw); if (i >= 0) tweens.splice(i, 1); };
    }
    function setView(v, ms) {
      const to = {}; for (const k of ['yaw', 'pitch', 'dist']) if (v[k] != null) to[k] = v[k];
      if (!ms || instant()) { viewTween = null; Object.assign(S, to); redraw(); return; }
      viewTween = { t0: performance.now(), ms, from: { yaw: S.yaw, pitch: S.pitch, dist: S.dist }, to }; kick();
    }
    function pulse(word) { glow[word] = performance.now(); kick(); }

    // Sizing, visibility and theme.
    function resize() {
      const w = el.clientWidth, h = Math.round(w * (typeof aspect === 'function' ? aspect(w) : aspect)), r = devicePixelRatio || 1;
      el.style.height = h + 'px'; el.width = Math.round(w * r); el.height = Math.round(h * r);
      ctx.setTransform(r, 0, 0, r, 0, 0); S.w = w; S.h = h; draw(performance.now());
    }
    const ro = new ResizeObserver(resize); ro.observe(el);
    const io = new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && (dirty || active())) kick(); else if (!visible && raf) { cancelAnimationFrame(raf); raf = 0; running = false; last = 0; } });
    io.observe(el);
    const mq = matchMedia('(prefers-color-scheme: dark)'), onTheme = () => redraw(); mq.addEventListener('change', onTheme);
    const fig = el.closest('figure'), onLink = e => pulse(e.detail); if (fig) fig.addEventListener('pe-link', onLink);

    // Interaction.
    const canTurn = o.draggable !== false;
    el.classList.add('pe3'); el.classList.toggle('pe3-static', !canTurn); el.style.touchAction = 'pan-y';
    if (canTurn && !el.hasAttribute('tabindex')) el.tabIndex = 0;
    const PMAX = Math.PI / 2 - .05;
    const turn = (dy, dp) => { viewTween = null; S.yaw += dy; S.pitch = clamp(S.pitch + dp, -PMAX, PMAX); redraw(); };
    let drag = null;
    const onDown = e => { if (!canTurn) return; drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); el.classList.add('pe3-drag'); };
    const onMove = e => { if (!drag) return; turn(-(e.clientX - drag.x) * .01, (e.clientY - drag.y) * .01); drag = { x: e.clientX, y: e.clientY }; };
    const onUp = () => { drag = null; el.classList.remove('pe3-drag'); };
    const onKey = e => {
      if (!canTurn) return;
      const d = { ArrowLeft: [-.15, 0], ArrowRight: [.15, 0], ArrowUp: [0, .15], ArrowDown: [0, -.15] }[e.key];
      if (d) { e.preventDefault(); turn(d[0], d[1]); }
    };
    el.addEventListener('pointerdown', onDown); el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp); el.addEventListener('pointercancel', onUp); el.addEventListener('keydown', onKey);

    function destroy() {
      dead = true; if (raf) cancelAnimationFrame(raf); tweens.length = 0; viewTween = null; for (const k in glow) delete glow[k]; ro.disconnect(); io.disconnect(); mq.removeEventListener('change', onTheme);
      if (fig) fig.removeEventListener('pe-link', onLink);
      el.removeEventListener('pointerdown', onDown); el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp); el.removeEventListener('pointercancel', onUp); el.removeEventListener('keydown', onKey);
    }
    return Object.assign(S, { project, setView, tween, redraw, destroy, pulse });
  }
  return { scene, surface, gaussian, rng, mix, instant };
})();
</script>

<style>
/* Page chrome: chapter rail with reading progress, a sticky bar below the rail breakpoint, figures as wide stages,
   and equation terms that light up their figure. Include once, anywhere in <main>, after nav.toc. */
:root { --rail-w:13.5rem; --measure:40rem; --stage:60rem; }
main { max-width:calc(var(--measure) + 6rem); margin:0 auto; padding:0 16px 5rem; }
main > *:not(figure):not(header):not(nav):not(section):not(.eq),
section.chapter > *:not(figure):not(.eq):not(.pe-wide) { max-width:var(--measure); }
figure.viz { margin:1.8rem 0; }
@media (max-width:520px) { figure.viz { margin:1.4rem -16px; padding:.7rem; border-radius:0; } }
h2[data-n]::before { content:attr(data-n); display:block; font-size:1rem; font-weight:400; color:var(--blue); margin-bottom:.35rem; }
@media (min-width:1180px) {
  main { max-width:var(--stage); margin-left:max(calc(var(--rail-w) + 3rem), calc((100vw - var(--stage)) / 2)); margin-right:auto; padding-left:0; }
}

.pe-rail { font:.86rem/1.35 system-ui, sans-serif; color:var(--muted); }
.pe-rail ol { list-style:none; margin:0; padding:0; }
.pe-rail a { color:inherit; text-decoration:none; display:flex; gap:.6rem; align-items:baseline; padding:.18rem 0; border-radius:4px; }
.pe-rail a:hover { color:var(--fg); }
.pe-rail a:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
.pe-rail .pe-act { font:1rem/1.2 KaTeX_Main, Charter, Georgia, serif; color:var(--fg); margin:1rem 0 .3rem; }
.pe-rail .pe-act:first-child { margin-top:0; }
.pe-rail .pe-act i { font-style:normal; color:var(--muted); margin-right:.35rem; }
.pe-rail li.pe-node { position:relative; padding-left:1.25rem; }
.pe-rail li.pe-node::before { content:""; position:absolute; left:.2rem; top:.5rem; width:.55rem; height:.55rem; border-radius:50%;
  border:1.5px solid var(--rule); background:var(--bg); z-index:1; transition:background .3s, border-color .3s; }
.pe-rail li.pe-node::after { content:""; position:absolute; left:calc(.475rem - 1px); top:1.05rem; bottom:-.55rem; width:2px;
  background:linear-gradient(var(--muted) calc(var(--p, 0) * 100%), var(--rule) 0); }
.pe-rail li.pe-node:last-child::after, .pe-rail li.pe-node.pe-end::after { display:none; }
.pe-rail li.pe-done::before { background:var(--muted); border-color:var(--muted); }
.pe-rail li.pe-done::after { --p:1; }
.pe-rail li.pe-cur::before { background:var(--blue); border-color:var(--blue); box-shadow:0 0 0 3px color-mix(in srgb, var(--blue) 30%, transparent); }
.pe-rail li.pe-cur > a { color:var(--fg); }
.pe-rail .pe-n { min-width:1.3em; font-variant-numeric:tabular-nums; }
.pe-rail .pe-extra { margin-top:1rem; padding-left:1.25rem; }
.pe-rail-title { font:1.05rem/1.25 KaTeX_Main, Charter, Georgia, serif; color:var(--fg); margin:0 0 1.2rem; display:block; text-decoration:none; }
.pe-bar { display:none; }
@media (min-width:1180px) {
  .pe-rail { position:fixed; top:0; bottom:0; left:0; width:var(--rail-w); padding:2rem 1rem 2rem 1.4rem; overflow-y:auto; border-right:1px solid var(--rule); background:var(--bg); z-index:20; }
}
@media (max-width:1179.98px) {
  .pe-bar { display:flex; position:sticky; top:0; z-index:30; align-items:center; gap:.7rem; height:2.7rem; padding:0 16px;
    background:color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter:blur(6px); border-bottom:1px solid var(--rule);
    font:.9rem system-ui, sans-serif; }
  .pe-bar-label { flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--fg); }
  .pe-bar-label b { color:var(--blue); margin-right:.45rem; font-variant-numeric:tabular-nums; }
  .pe-bar button { font:inherit; color:var(--fg); background:var(--panel2, var(--panel)); border:1px solid var(--rule); border-radius:999px; padding:.25rem .8rem; cursor:pointer; }
  .pe-bar button:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
  .pe-bar-progress { position:absolute; left:0; bottom:-1px; height:2px; width:100%; background:var(--blue); transform-origin:0 50%; transform:scaleX(var(--g, 0)); }
  .pe-rail { display:none; }
  body.pe-rail-open .pe-rail { display:block; position:fixed; top:2.7rem; left:0; right:0; max-height:72vh; overflow-y:auto; z-index:29;
    background:var(--bg); border-bottom:1px solid var(--rule); padding:1rem 16px 1.2rem; }
}
body.pe-watch .pe-rail, body.pe-watch .pe-bar { display:none !important; }
body.pe-watch main { margin-left:auto; margin-right:auto; }

/* Equation terms that name part of a figure. */
.eq[data-viz] [data-link].pe-term { cursor:pointer; border-radius:3px; transition:background-color .2s; }
.eq[data-viz] [data-link].pe-term:hover, .eq[data-viz] [data-link].pe-term:focus-visible { background:color-mix(in srgb, var(--fg) 12%, transparent); outline:none; }
.pe-lit { background:color-mix(in srgb, var(--yellow) 22%, transparent) !important; border-radius:3px; transition:background-color .2s; }
@media (prefers-reduced-motion: reduce) { .pe-rail li.pe-node::before, .pe-term, .pe-lit { transition:none !important; } }
</style>
<script>
// Runs after parsing: the chapters come after this script in the page.
addEventListener('DOMContentLoaded', () => {
  const toc = document.querySelector('nav.toc');
  const sections = [...document.querySelectorAll('section.chapter[id]')];
  const byId = Object.fromEntries(sections.map(s => [s.id, s]));
  // Rail items come from nav.toc (acts are li.act, chapters are numbered li > a); without a toc, from the h2s.
  const items = [];
  if (toc) {
    toc.querySelectorAll('ol > li').forEach(li => {
      if (li.classList.contains('act')) { items.push({ act: li.textContent.trim() }); return; }
      const a = li.querySelector('a[href^="#"]'); if (!a || !byId[a.getAttribute('href').slice(1)]) return;
      const full = a.textContent.trim(), paren = /\(([^()]*)\)\s*$/.exec(full);
      items.push({ id: a.getAttribute('href').slice(1), label: a.dataset.rail || (paren ? paren[1] : full), full });
    });
  } else sections.forEach(s => items.push({ id: s.id, label: s.querySelector('h2').textContent.trim() }));
  let n = 0; items.forEach(it => { if (it.id) it.n = ++n; });
  const numbered = new Set(items.filter(i => i.id).map(i => i.id));
  const extras = sections.filter(s => !numbered.has(s.id));
  items.forEach(it => { if (it.id) byId[it.id].querySelector('h2').dataset.n = it.n; });

  const rail = document.createElement('nav'); rail.className = 'pe-rail'; rail.id = 'pe-rail'; rail.setAttribute('aria-label', 'Chapters and reading progress');
  const title = document.querySelector('h1');
  const ol = document.createElement('ol');
  const actName = t => { const m = /^Act ([IVXLC]+):?\s*(.*)$/i.exec(t); return m ? `<i>${m[1]}</i>${m[2].charAt(0).toUpperCase() + m[2].slice(1)}` : t; };
  items.forEach((it, k) => {
    const li = document.createElement('li');
    if (it.act) { li.className = 'pe-act'; li.innerHTML = actName(it.act); }
    else {
      li.className = 'pe-node'; li.dataset.id = it.id; if (!items[k + 1] || items[k + 1].act) li.classList.add('pe-end');
      li.innerHTML = `<a href="#${it.id}" title="${(it.full || it.label).replace(/"/g, '&quot;')}"><span class="pe-n">${it.n}</span><span>${it.label}</span></a>`;
    }
    ol.appendChild(li);
  });
  rail.innerHTML = title ? `<a class="pe-rail-title" href="#top">${title.textContent}</a>` : '';
  rail.appendChild(ol);
  if (extras.length) { const ex = document.createElement('ol'); ex.className = 'pe-extra';
    extras.forEach(s => { const li = document.createElement('li'); li.innerHTML = `<a href="#${s.id}">${s.querySelector('h2').textContent}</a>`; ex.appendChild(li); }); rail.appendChild(ex); }
  if (!document.getElementById('top')) { const t = document.createElement('span'); t.id = 'top'; document.body.prepend(t); }
  const bar = document.createElement('div'); bar.className = 'pe-bar';
  bar.innerHTML = '<span class="pe-bar-label" aria-live="polite"></span><button type="button" id="pe-rail-toggle" aria-expanded="false" aria-controls="pe-rail">Chapters</button><span class="pe-bar-progress"></span>';
  document.body.insertBefore(bar, document.body.firstChild.nextSibling);
  document.body.insertBefore(rail, bar.nextSibling);
  const toggle = bar.querySelector('button'), label = bar.querySelector('.pe-bar-label');
  const setOpen = o => { document.body.classList.toggle('pe-rail-open', o); toggle.setAttribute('aria-expanded', String(o)); };
  toggle.addEventListener('click', () => setOpen(!document.body.classList.contains('pe-rail-open')));
  rail.addEventListener('click', e => { if (e.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') setOpen(false); });

  const nodes = [...rail.querySelectorAll('li.pe-node')];
  let queued = false;
  function spy() {
    queued = false;
    const line = innerHeight * .35; let cur = -1;
    nodes.forEach((li, k) => { if (byId[li.dataset.id].getBoundingClientRect().top <= line) cur = k; });
    nodes.forEach((li, k) => {
      li.classList.toggle('pe-done', k < cur); li.classList.toggle('pe-cur', k === cur);
      if (k === cur) { const r = byId[li.dataset.id].getBoundingClientRect(); li.style.setProperty('--p', Math.min(1, Math.max(0, (line - r.top) / r.height)).toFixed(3)); }
      else li.style.removeProperty('--p');
    });
    const it = cur >= 0 ? items.find(i => i.id === nodes[cur].dataset.id) : null;
    label.innerHTML = it ? `<b>${it.n}</b>${it.label}` : (title ? title.textContent : '');
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.setProperty('--g', max > 0 ? (scrollY / max).toFixed(4) : 0);
  }
  addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(spy); } }, { passive: true });
  addEventListener('resize', spy); spy();

  // Hover, focus or tap an equation term: light it, light the figure parts with the same data-link, and pulse the canvas.
  document.querySelectorAll('.eq[data-viz]').forEach(eq => {
    const fig = document.getElementById(eq.dataset.viz); if (!fig) return;
    eq.querySelectorAll('[data-link]').forEach(t => {
      t.classList.add('pe-term'); t.tabIndex = 0; t.setAttribute('role', 'button');
      t.setAttribute('aria-label', 'Show ' + t.dataset.link.replace(/\s+/g, ' and ') + ' in the figure');
      let last = 0;
      const fire = () => {
        const now = performance.now(); if (now - last < 600) return; last = now;
        const lit = [t];
        t.dataset.link.split(/\s+/).forEach(l => { fig.querySelectorAll(`[data-link~="${l}"]`).forEach(x => lit.push(x)); fig.dispatchEvent(new CustomEvent('pe-link', { detail: l })); });
        lit.forEach(x => x.classList.add('pe-lit')); setTimeout(() => lit.forEach(x => x.classList.remove('pe-lit')), 1100);
      };
      t.addEventListener('mouseenter', fire); t.addEventListener('focus', fire); t.addEventListener('click', fire);
      t.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); last = 0; fire(); } });
    });
  });
});
</script>

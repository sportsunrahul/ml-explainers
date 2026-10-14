/* Long-page twin of the atlas shell's scene helpers. A scene is SCENES.id = ctx => handle, written once
   and mounted in both pages. Here ctx.stage is the figure's .stage div, ctx.article is the figure, ctx.hud
   writes the figure's readout (the badge moves to the figure's corner). Keep these names and signatures
   identical to assets/atlas/shell.html. */
const SCENES = {};
const S3ASPECT = stage => w => Math.max(.3, (stage.clientHeight || 1) / Math.max(w, 1));
const mkCanvas = (stage, cls) => { const el = document.createElement('canvas'); if (cls) el.className = cls; stage.appendChild(el); return el; };
const listen = (el, ev, fn) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); };
const press = (btns, test) => btns.forEach(b => b.setAttribute('aria-pressed', String(test(b))));
const badge = (kind, text) => `<span class="badge ${kind}">${{ toy: 'Toy model', computed: 'Computed from the paper', sourced: 'Sourced numbers' }[kind]}</span><br>${text}`;
const mid = (pts) => pts.reduce((s, p) => [s[0] + p[0] / pts.length, s[1] + p[1] / pts.length, s[2] + p[2] / pts.length], [0, 0, 0]);
const SCENEMOUNT = (() => {
  const live = new Map();
  function mount(fig) {
    const name = fig.dataset.scene, stage = fig.querySelector('.stage'), out = fig.querySelector('.readout');
    let slot = fig.querySelector(':scope > .badge-slot');
    if (!slot) { slot = document.createElement('div'); slot.className = 'badge-slot'; fig.prepend(slot); }
    const old = live.get(fig); if (old && old.destroy) old.destroy(); stage.replaceChildren();
    let lastText = null;
    const hud = html => { const t = document.createElement('div'); t.innerHTML = html; const b = t.querySelector('.badge');
      if (b) { b.remove(); if (slot.innerHTML !== b.outerHTML) slot.innerHTML = b.outerHTML; }
      const txt = t.innerHTML.replace(/^\s*<br>/, ''); if (out && txt !== lastText) { out.innerHTML = txt; lastText = txt; } };
    const ctx = { stage, article: fig, hud, q: s => fig.querySelector(s), qa: s => [...fig.querySelectorAll(s)] };
    const h = SCENES[name] ? SCENES[name](ctx) : null; live.set(fig, h);
    const tabs = ctx.qa('[role="tab"][data-step]');
    if (tabs.length) setStep(fig, +(fig.dataset.step || 0));
    return h;
  }
  function setStep(fig, n) {
    fig.dataset.step = n;
    fig.querySelectorAll('[role="tab"][data-step]').forEach(b => b.setAttribute('aria-selected', String(+b.dataset.step === n)));
    fig.querySelectorAll('.step-text').forEach(d => d.classList.toggle('on', +d.dataset.step === n));
    const h = live.get(fig); if (h && h.step) h.step(n);
  }
  document.addEventListener('click', e => {
    const tab = e.target.closest('figure.viz[data-scene] [role="tab"][data-step]');
    if (tab) { setStep(tab.closest('figure'), +tab.dataset.step); return; }
    const rp = e.target.closest('figure.viz[data-scene] [data-replay]');
    if (rp) { const h = live.get(rp.closest('figure')); if (h && h.replay) h.replay(); return; }
    const rs = e.target.closest('figure.viz[data-scene] [data-cue-reset]');
    if (rs) { const f = rs.closest('figure'); f.dataset.step = 0;
      // scenes read pressed buttons and slider values back on mount (so state survives an atlas theme remount); a reset clears them first
      f.querySelectorAll('[aria-pressed]').forEach(b => b.removeAttribute('aria-pressed'));
      f.querySelectorAll('input[type=range]').forEach(i => { i.value = i.defaultValue; });
      mount(f); }
  });
  function all() { document.querySelectorAll('figure.viz[data-scene]').forEach(mount); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', all); else queueMicrotask(all);
  return { mount, setStep, live };
})();

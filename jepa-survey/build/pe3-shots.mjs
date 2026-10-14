// node ~/paper-explainers/jepa-survey/build/pe3-shots.mjs   (run with cwd = ~/paper-explainers/.node)
// Inlines assets/pe3.js into pe3-test.html, then checks the page under four contexts and writes PNGs to pe3-shots/.
import fs from 'fs'; import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const DIR = path.dirname(new URL(import.meta.url).pathname), OUT = path.join(DIR, 'pe3-shots');
const ASSET = path.join(process.env.HOME, '.claude/skills/paper-explainer/assets/pe3.js');
fs.mkdirSync(OUT, { recursive: true });
const html = fs.readFileSync(path.join(DIR, 'pe3-test.src.html'), 'utf8').replace('<!--PE3-->', fs.readFileSync(ASSET, 'utf8'));
const FILE = path.join(DIR, 'pe3-test.html'); fs.writeFileSync(FILE, html);
const URL_ = 'file://' + FILE;

const CTX = [
  ['dark', { viewport: { width: 1280, height: 800 }, colorScheme: 'dark' }],
  ['light', { viewport: { width: 1280, height: 800 }, colorScheme: 'light' }],
  ['mobile', { viewport: { width: 375, height: 812 }, colorScheme: 'dark' }],
  ['reduced', { viewport: { width: 1280, height: 800 }, colorScheme: 'dark', reducedMotion: 'reduce' }],
];
// [button, pre-clicks that set up a state worth leaving]
const BTNS = [['btn-collapse', []], ['btn-isotropic', ['btn-collapse']], ['btn-reset', ['btn-collapse']], ['btn-pulse-emb', []], ['btn-replan', []]];
const FIGS = ['viz-emb', 'viz-energy', 'viz-plan'];
const fails = [];
const b = await chromium.launch();

for (const [name, opt] of CTX) {
  const ctx = await b.newContext(opt), p = await ctx.newPage();
  p.on('pageerror', e => fails.push(`${name}: pageerror ${e.message}`));
  p.on('console', m => { if (m.type() === 'error') fails.push(`${name}: console ${m.text()}`); });
  const load = async () => { await p.goto(URL_); await p.waitForTimeout(400); };
  const canvasOf = fig => `#${fig} canvas`;
  const dataURL = fig => p.$eval(canvasOf(fig), c => c.toDataURL());
  await load();
  await p.screenshot({ path: `${OUT}/page-${name}.png`, fullPage: true });
  for (const f of FIGS) { await p.locator('#' + f).scrollIntoViewIfNeeded(); await p.waitForTimeout(150); await p.locator('#' + f).screenshot({ path: `${OUT}/${f}-${name}.png` }); }
  if (name === 'mobile') {
    const sw = await p.evaluate(() => document.documentElement.scrollWidth);
    if (sw > 375) fails.push(`mobile: horizontal overflow, scrollWidth ${sw}`);
  }

  // Idempotence: one click vs a second click mid-animation must end on the same pixels.
  for (const [id, pre] of BTNS) {
    const fig = id === 'btn-replan' ? 'viz-plan' : 'viz-emb';
    const run = async twice => {
      await load(); await p.locator('#' + fig).scrollIntoViewIfNeeded();
      for (const q of pre) { await p.click('#' + q); await p.waitForTimeout(1200); }
      await p.click('#' + id);
      if (twice) { await p.waitForTimeout(300); await p.click('#' + id); }
      await p.waitForTimeout(1600); return dataURL(fig);
    };
    const a = await run(false), c = await run(true);
    if (a !== c) fails.push(`${name}: ${id} not idempotent`);
    if (name === 'dark') { await p.locator('#' + fig).screenshot({ path: `${OUT}/${fig}-after-${id}.png` }); }
  }

  // Drag rotates the first canvas.
  await load(); await p.locator('#viz-emb').scrollIntoViewIfNeeded();
  const before = await dataURL('viz-emb'), box = await p.locator(canvasOf('viz-emb')).boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await p.mouse.move(cx, cy); await p.mouse.down(); await p.mouse.move(cx + 80, cy + 30, { steps: 8 }); await p.mouse.up(); await p.waitForTimeout(250);
  if (await dataURL('viz-emb') === before) fails.push(`${name}: drag did not change pixels`);
  if (name === 'dark') await p.locator('#viz-emb').screenshot({ path: `${OUT}/viz-emb-dragged.png` });

  // Keyboard rotate.
  await load(); await p.locator('#viz-emb').scrollIntoViewIfNeeded();
  const b0 = await dataURL('viz-emb'); await p.focus(canvasOf('viz-emb')); await p.keyboard.press('ArrowLeft'); await p.waitForTimeout(200);
  if (await dataURL('viz-emb') === b0) fails.push(`${name}: arrow key did not change pixels`);

  // pe-link pulse, shot 0.4 s later.
  await load(); await p.locator('#viz-emb').scrollIntoViewIfNeeded();
  const p0 = await dataURL('viz-emb');
  await p.$eval('#viz-emb', f => f.dispatchEvent(new CustomEvent('pe-link', { detail: 'emb' }))); await p.waitForTimeout(400);
  await p.locator('#viz-emb').screenshot({ path: `${OUT}/viz-emb-pulse-${name}.png` });
  if (await dataURL('viz-emb') === p0) fails.push(`${name}: pulse did not change pixels at 0.4 s`);
  await p.waitForTimeout(1200);
  if (await dataURL('viz-emb') !== p0) fails.push(`${name}: pulse did not return to the base frame`);

  // Render on demand: no rAF calls once nothing moves.
  const rafCount = () => p.evaluate(() => new Promise(r => { let n = 0; const o = window.requestAnimationFrame;
    window.requestAnimationFrame = f => { n++; return o.call(window, f); }; setTimeout(() => { window.requestAnimationFrame = o; r(n); }, 400); }));
  const idle = async label => { const n = await rafCount(); if (n) fails.push(`${name}: loop not idle after ${label} (${n} rAF calls in 400 ms)`); };
  await load(); await idle('load');
  await p.click('#btn-collapse'); await p.waitForTimeout(1200); await idle('collapse');
  await p.$eval('#viz-emb', f => f.dispatchEvent(new CustomEvent('pe-link', { detail: 'emb' }))); await p.waitForTimeout(1500); await idle('pulse');
  await p.$eval('#viz-emb', f => f.dispatchEvent(new CustomEvent('pe-link', { detail: 'nope' }))); await p.waitForTimeout(1500); await idle('unmatched pulse');
  if (name === 'reduced') { await p.evaluate(() => { window.pe3scenes.emb.autoRotate = .5; window.pe3scenes.emb.redraw(); }); await p.waitForTimeout(300); await idle('autoRotate under reduced motion'); }

  // Theme change in place must match a fresh context of the other theme.
  if (name === 'dark') {
    await load(); await p.emulateMedia({ colorScheme: 'light' }); await p.waitForTimeout(300);
    const live = {}; for (const f of FIGS) { await p.locator('#' + f).scrollIntoViewIfNeeded(); await p.waitForTimeout(200); live[f] = await dataURL(f); }
    const c2 = await b.newContext({ ...opt, colorScheme: 'light' }), p2 = await c2.newPage(); await p2.goto(URL_); await p2.waitForTimeout(400);
    for (const f of FIGS) if (await p2.$eval(`#${f} canvas`, c => c.toDataURL()) !== live[f]) fails.push(`dark->light switch: ${f} differs from a fresh light page`);
    await c2.close();
  }

  // Range input drives the marker.
  await load(); await p.locator('#viz-energy').scrollIntoViewIfNeeded();
  const e0 = await dataURL('viz-energy');
  await p.$eval('#energy-t', i => { i.value = '.9'; i.dispatchEvent(new Event('input')); }); await p.waitForTimeout(200);
  if (await dataURL('viz-energy') === e0) fails.push(`${name}: range input did not move the marker`);
  if (name === 'dark') await p.locator('#viz-energy').screenshot({ path: `${OUT}/viz-energy-t09.png` });
  await ctx.close();
}
await b.close();
if (fails.length) { console.log('FAIL\n' + fails.join('\n')); process.exit(1); }
console.log('OK: all contexts clean; shots in ' + OUT);

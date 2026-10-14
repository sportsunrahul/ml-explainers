// node shots.mjs <html> <outdir> [selector=figure.viz] [widths=1280,375] : reduced-motion screenshots of each figure, both themes; prints console errors.
import { chromium } from 'playwright';
import fs from 'fs';
const [,, html, out, sel = 'figure.viz', ws = '1280,375'] = process.argv;
fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch();
for (const w of ws.split(',').map(Number)) for (const scheme of ['dark', 'light']) {
  const p = await b.newPage({ viewport: { width: w, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' });
  const errs = []; p.on('console', m => m.type() === 'error' && errs.push(m.text())); p.on('pageerror', e => errs.push('PAGEERROR ' + e.message));
  await p.goto('file://' + html); await p.waitForTimeout(600);
  const figs = await p.$$(sel);
  for (const f of figs) { const id = await f.getAttribute('id'); await f.scrollIntoViewIfNeeded(); await p.waitForTimeout(150); await f.screenshot({ path: `${out}/${id}-${w}-${scheme}.png` }); }
  const ov = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  console.log(w, scheme, figs.length, 'figs', 'overflow', ov, errs.length ? 'ERRORS ' + errs.join(' | ') : 'no errors');
  await p.close();
}
await b.close();

// node qa.mjs <page.html> <outdir>   (cwd = ~/paper-explainers/.node)
// Screenshots and checks for an "explained in 3D" page. Prints one JSON summary line; exit 1 on errors or overflow.
import fs from 'fs'; import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2);
for (const d of ['top', 'figs', 'links']) fs.mkdirSync(path.join(out, d), { recursive: true });
const b = await chromium.launch();
const sum = { errors: [], overflow: [], nav: {}, figs: 0, links: 0, katex: 0 };
const open = async (w, cs, rm = 'no-preference') => {
  const p = await (await b.newContext({ viewport: { width: w, height: w < 600 ? 812 : 900 }, colorScheme: cs, reducedMotion: rm })).newPage();
  p.on('pageerror', e => sum.errors.push(`${w} ${cs}: ${e.message}`));
  p.on('console', m => { if (m.type() === 'error') sum.errors.push(`${w} ${cs}: ${m.text()}`); });
  await p.goto('file://' + file, { waitUntil: 'networkidle' });
  await p.evaluate(() => document.documentElement.style.scrollBehavior = 'auto');
  return p;
};
for (const [w, cs] of [[1440, 'dark'], [1440, 'light'], [1280, 'dark'], [768, 'dark'], [375, 'dark'], [375, 'light']]) {
  const p = await open(w, cs);
  if (w === 1440 && cs === 'dark') for (const ms of [1500, 3200, 4800]) { await p.waitForTimeout(ms === 1500 ? 1500 : 1650); if (await p.$('#f-hero')) await p.locator('#f-hero').screenshot({ path: `${out}/top/hero-mid-${ms}.png` }); }
  await p.waitForTimeout(7000);
  const info = await p.evaluate(() => ({ over: document.documentElement.scrollWidth > innerWidth + 1, collapsed: document.querySelector('nav.top')?.classList.contains('nav-collapsed'), katex: document.querySelectorAll('.katex').length }));
  if (info.over) sum.overflow.push(`${w} ${cs}`); sum.nav[`${w}`] = info.collapsed ? 'menu' : 'links'; sum.katex = info.katex;
  await p.screenshot({ path: `${out}/top/top-${w}-${cs}.png` });
  if (await p.$('#f-hero')) await p.locator('#f-hero').screenshot({ path: `${out}/top/hero-${w}-${cs}.png` });
  // Mid-page: the bar shows the current section and progress.
  await p.evaluate(() => scrollTo(0, document.documentElement.scrollHeight * .45)); await p.waitForTimeout(500);
  await p.screenshot({ path: `${out}/top/mid-${w}-${cs}.png` });
  if (await p.$('.nav-menu') && info.collapsed) { await p.click('.nav-menu'); await p.waitForTimeout(250); await p.screenshot({ path: `${out}/top/menu-${w}-${cs}.png` }); await p.keyboard.press('Escape'); }
  if (w === 1280 && cs === 'dark') {
    const ids = await p.$$eval('figure.fig[id]', fs => fs.map(f => f.id));
    for (const id of ids) { await p.$eval('#' + id, f => f.scrollIntoView({ block: 'center' })); await p.waitForTimeout(2200);
      await p.locator('#' + id).screenshot({ path: `${out}/figs/${id}.png` }); sum.figs++; }
    const terms = await p.$$eval('.eq[data-viz] [data-link]', ts => ts.map((t, i) => { t.dataset.qa = i; return [i, t.closest('.eq').id, t.closest('.eq').dataset.viz, t.dataset.link]; }));
    for (const [i, eq, fig, word] of terms) {
      await p.$eval('#' + fig, f => f.scrollIntoView({ block: 'center' })); await p.waitForTimeout(1500);
      await p.$eval(`[data-qa="${i}"]`, t => t.focus({ preventScroll: true })); await p.waitForTimeout(520);
      await p.locator('#' + fig).screenshot({ path: `${out}/links/${eq}-${word}-${i}.png` }); sum.links++;
      await p.$eval(`[data-qa="${i}"]`, t => t.blur()); await p.waitForTimeout(700);
    }
  }
  await p.close();
}
const r = await open(1440, 'dark', 'reduce'); await r.waitForTimeout(1500);
if (await r.$('#f-hero')) await r.locator('#f-hero').screenshot({ path: `${out}/top/hero-reduced.png` });
await r.close(); await b.close();
console.log(JSON.stringify(sum));
process.exit(sum.errors.length || sum.overflow.length ? 1 : 0);

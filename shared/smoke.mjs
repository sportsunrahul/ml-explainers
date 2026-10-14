// node smoke.mjs <page.html>...: click every button and move every input in every figure; print page errors per file.
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const b = await chromium.launch();
for (const file of process.argv.slice(2)) {
  const p = await (await b.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark' })).newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto('file://' + file, { waitUntil: 'networkidle' }); await p.waitForTimeout(1500);
  await p.evaluate(() => document.documentElement.style.scrollBehavior = 'auto');
  const n = await p.$$eval('figure.fig', fs => fs.length);
  let acts = 0;
  for (let i = 0; i < n; i++) {
    await p.evaluate(i => document.querySelectorAll('figure.fig')[i].scrollIntoView({ block: 'center' }), i); await p.waitForTimeout(1200);
    acts += await p.evaluate(async i => {
      const f = document.querySelectorAll('figure.fig')[i]; let k = 0; const wait = ms => new Promise(r => setTimeout(r, ms));
      for (const el of f.querySelectorAll('button')) { el.click(); k++; await wait(120); }
      for (const el of f.querySelectorAll('input[type=range]')) { for (const v of [el.min, el.max, (+el.min + +el.max) / 2]) { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); await wait(80); } k++; }
      for (const el of f.querySelectorAll('input[type=checkbox]')) { el.click(); await wait(120); el.click(); k++; }
      return k; }, i);
    await p.waitForTimeout(600);
  }
  // Equation terms: fire each once.
  for (const t of await p.$$('.eq[data-viz] [data-link]')) { await t.focus(); await p.waitForTimeout(150); }
  await p.waitForTimeout(1200);
  console.log(JSON.stringify({ file: path.basename(file), figures: n, actions: acts, errors: [...new Set(errs)] }));
  await p.close();
}
await b.close();

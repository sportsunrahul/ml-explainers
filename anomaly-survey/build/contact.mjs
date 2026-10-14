// Screenshots every figure.viz in the assembled page (1280 dark, 1280 light, 375 dark) and writes contact-sheet pages.
import path from 'path'; import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2); fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch();
for (const [name, vp, cs] of [['dark', { width: 1280, height: 900 }, 'dark'], ['light', { width: 1280, height: 900 }, 'light'], ['mobile', { width: 375, height: 812 }, 'dark']]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: cs, reducedMotion: 'reduce' })).newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto('file://' + file); await p.waitForTimeout(800);
  const ids = await p.$$eval('figure.viz[id], figure.archflow[id]', fs => fs.map(f => f.id));
  const shots = [];
  for (const id of ids) { const f = p.locator('#' + id); await f.scrollIntoViewIfNeeded(); await p.waitForTimeout(250);
    const fn = `${name}-${id}.png`; await f.screenshot({ path: path.join(out, fn) }); shots.push([id, fn]); }
  const w = name === 'mobile' ? 380 : 640;
  for (let i = 0; i < shots.length; i += 6) fs.writeFileSync(path.join(out, `sheet-${name}-${i / 6}.html`),
    `<body style="margin:0;background:#777;display:flex;flex-wrap:wrap;gap:6px;width:${w * 3 + 12}px">` + shots.slice(i, i + 6).map(([id, fn]) =>
      `<figure style="margin:0;width:${w}px"><img src="${fn}" style="width:${w}px"><figcaption style="font:12px sans-serif;color:#fff">${id}</figcaption></figure>`).join('') + '</body>');
  console.log(name, ids.length, 'figures', errs.length ? 'ERRORS: ' + errs.slice(0, 3).join(' | ') : 'no errors');
}
await b.close();

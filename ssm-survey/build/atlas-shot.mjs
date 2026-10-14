import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out, chapter, wait = '7000', clicks = ''] = process.argv.slice(2);
const b = await chromium.launch();
for (const [name, vp, cs] of [['d-dark', { width: 1440, height: 900 }, 'dark'], ['d-light', { width: 1440, height: 900 }, 'light'], ['m-dark', { width: 390, height: 844 }, 'dark']]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: cs })).newPage();
  p.on('pageerror', e => console.log('ERR', name, e.message)); p.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', name, m.text()); });
  await p.goto('file://' + file + '#' + chapter); await p.waitForTimeout(+wait);
  await p.screenshot({ path: `${out}/${name}-${chapter}-0.png` });
  let i = 1;
  for (const sel of clicks.split('|').filter(Boolean)) { await p.click(sel); await p.waitForTimeout(1500); await p.screenshot({ path: `${out}/${name}-${chapter}-${i++}.png` }); }
}
await b.close();

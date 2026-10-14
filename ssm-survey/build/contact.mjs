import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out, scheme = 'dark', width = '1280'] = process.argv.slice(2);
const b = await chromium.launch();
const p = await (await b.newContext({ viewport: { width: +width, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' })).newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
await p.goto('file://' + file); await p.waitForTimeout(800);
const ids = await p.$$eval('figure.viz', fs => fs.map(f => f.id));
for (const id of ids) { const f = p.locator('#' + id); await f.scrollIntoViewIfNeeded(); await p.waitForTimeout(250); await f.screenshot({ path: `${out}/${scheme}-${width}-${id}.png` }); }
console.log(ids.length, 'figures', errs.length ? errs : 'no errors');
await b.close();

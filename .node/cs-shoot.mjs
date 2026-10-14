import { chromium } from 'playwright';
const [page_, out, width, scheme] = process.argv.slice(2);
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: +width, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' });
const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type() === 'error' && errs.push(m.text()));
await p.goto('file://' + page_); await p.waitForTimeout(800);
const ids = await p.$$eval('figure.viz', fs => fs.map(f => f.id));
for (const id of ids) { const el = await p.$('#' + id); await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(250); await el.screenshot({ path: `${out}/${width}-${scheme}-${id}.png` }); }
console.log(ids.length, 'figures', errs.length ? 'ERRORS ' + errs.join(' | ') : 'no errors');
await b.close();

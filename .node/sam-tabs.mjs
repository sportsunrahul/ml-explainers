// node sam-tabs.mjs <html> <outdir> <figId> <btnIds,comma> [width] [scheme]
import { chromium } from 'playwright';
const [,, html, out, fig, btns, w = '1280', scheme = 'dark'] = process.argv;
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: +w, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' });
const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type() === 'error' && errs.push(m.text()));
await p.goto('file://' + html); await p.waitForTimeout(500);
for (const id of btns.split(',')) { await p.click('#' + id); await p.waitForTimeout(300); await p.locator('#' + fig).screenshot({ path: `${out}/${fig}-${id}-${w}-${scheme}.png` }); }
console.log(errs.length ? errs : 'ok'); await b.close();

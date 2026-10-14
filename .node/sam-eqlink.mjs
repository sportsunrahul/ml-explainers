// node sam-eqlink.mjs <html> <outdir> <eqId>: hover each [data-link] term in the equation, shoot its figure 0.5 s later.
import { chromium } from 'playwright';
const [,, html, out, eq] = process.argv;
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark' });
const errs = []; p.on('pageerror', e => errs.push(e.message));
await p.goto('file://' + html); await p.waitForTimeout(400);
const viz = await p.getAttribute('#' + eq, 'data-viz'); const terms = await p.$$(`#${eq} [data-link]`);
for (const t of terms) { const k = await t.getAttribute('data-link'); await t.scrollIntoViewIfNeeded(); await t.hover(); await p.waitForTimeout(550); await p.locator('#' + viz).screenshot({ path: `${out}/eqlink-${eq}-${k}.png` }); }
console.log(terms.length, 'terms', errs.length ? errs : 'ok'); await b.close();

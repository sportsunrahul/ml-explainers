// node figs.mjs <html> <outdir> <width> <scheme>
import path from 'path'; import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out, w, cs] = process.argv.slice(2); fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch(); const p = await (await b.newContext({ viewport: { width: +w, height: 900 }, colorScheme: cs })).newPage();
p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('file://' + file); await p.waitForTimeout(800);
const ids = await p.$$eval('figure.viz[id]', fs => fs.map(f => f.id));
for (const id of ids) { const f = p.locator('#' + id); await f.scrollIntoViewIfNeeded(); await p.waitForTimeout(150); await f.screenshot({ path: `${out}/${id}.png` }); }
// contact sheets, 6 per sheet
const imgs = ids.map(id => ({ id, src: 'data:image/png;base64,' + fs.readFileSync(`${out}/${id}.png`).toString('base64') }));
const q = await (await b.newContext({ viewport: { width: 1800, height: 1000 } })).newPage();
for (let s = 0; s * 6 < imgs.length; s++) {
  const html = `<body style="margin:0;background:#888;display:grid;grid-template-columns:repeat(3,1fr);gap:6px;padding:6px;font:14px sans-serif">${imgs.slice(s * 6, s * 6 + 6).map(i => `<div><div>${i.id}</div><img src="${i.src}" style="width:100%"></div>`).join('')}</body>`;
  await q.setContent(html); await q.waitForTimeout(200); await q.screenshot({ path: `${out}/sheet-${s}.png`, fullPage: true });
}
console.log(ids.length, 'figures');
await b.close();

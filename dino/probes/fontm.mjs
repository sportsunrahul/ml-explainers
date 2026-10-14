import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const b = await chromium.launch(); const p = await b.newPage();
await p.goto('file:///Users/rahulmishra/paper-explainers/dino/dino-explained.html', { waitUntil: 'networkidle' });
await p.evaluate(() => document.fonts.ready);
const r = await p.evaluate(() => { const m = (t, fv) => { const s = document.createElement('span'); s.style.cssText = 'font-family:"Schibsted Grotesk";font-size:100px;position:absolute;white-space:pre;font-variant-numeric:' + fv; s.textContent = t; document.body.appendChild(s); const w = s.getBoundingClientRect().width; s.remove(); return Math.round(w); };
  return { normal: { '1111': m('1111', 'normal'), '0000': m('0000', 'normal'), '....': m('....', 'normal') }, tab: { '1111': m('1111', 'tabular-nums'), '0000': m('0000', 'tabular-nums'), '....': m('....', 'tabular-nums') }, loaded: document.fonts.check('600 16px "Schibsted Grotesk"') }; });
console.log(JSON.stringify(r)); await b.close();

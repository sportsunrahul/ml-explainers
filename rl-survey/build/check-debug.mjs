// node check-page.mjs <projectDir> [--narrated]   (run with cwd = ~/paper-explainers/.node)
// Exit 1 on any failure. Screenshots go to <projectDir>/build/shots/.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const require = createRequire(path.join(process.cwd(), 'x.js'));
const { chromium } = require('playwright');

const dir = path.resolve(process.argv[2] || '.');
const narrated = process.argv.includes('--narrated');
const file = path.join(dir, narrated ? 'explainer.narrated.html' : 'explainer.html');
const shots = path.join(dir, 'build', 'shots');
fs.mkdirSync(shots, { recursive: true });

const fails = [];
const fail = m => fails.push(m);

const scriptDir = path.join(dir, 'scripts');
const cues = fs.existsSync(scriptDir)
  ? fs.readdirSync(scriptDir).filter(f => f.endsWith('.md')).flatMap(f =>
      [...fs.readFileSync(path.join(scriptDir, f), 'utf8').matchAll(/\[\[cue:([\w-]+)\]\]/g)].map(m => ({ f, id: m[1] })))
  : [];
const claimsFile = path.join(dir, 'notes', 'claims.json');
const claims = fs.existsSync(claimsFile)
  ? new Map(JSON.parse(fs.readFileSync(claimsFile, 'utf8')).map(c => [c.id, c])) : null;
const norm = x => String(x).toLowerCase().replace(/\s+/g, ' ').trim();

const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
const configs = [
  { name: 'desktop-dark', viewport: { width: 1280, height: 800 }, colorScheme: 'dark' },
  { name: 'desktop-light', viewport: { width: 1280, height: 800 }, colorScheme: 'light' },
  { name: 'mobile-dark', viewport: { width: 375, height: 812 }, colorScheme: 'dark' },
  { name: 'mobile-light', viewport: { width: 375, height: 812 }, colorScheme: 'light' },
  { name: 'reduced-motion', viewport: { width: 1280, height: 800 }, colorScheme: 'dark', reducedMotion: 'reduce' },
];

let review = [];
for (const [ci, cfg] of configs.entries()) {
  const ctx = await browser.newContext({ viewport: cfg.viewport, colorScheme: cfg.colorScheme, reducedMotion: cfg.reducedMotion || 'no-preference' });
  const page = await ctx.newPage();
  page.on('console', m => { if (m.type() === 'error') fail(`[${cfg.name}] console error: ${m.text()}`); });
  page.on("pageerror", e => fail(`[${cfg.name}] page exception: ${e.message}\n${(e.stack||"").split("\n").slice(0,5).join("\n")}`));
  await page.goto('file://' + file);
  await page.waitForTimeout(300);

  if (ci === 0) {
    const s = await page.evaluate(() => {
      const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
      const dup = ids.filter((x, i) => ids.indexOf(x) !== i);
      const badCh = [...document.querySelectorAll('section.chapter')]
        .filter(sec => !sec.id || sec.firstElementChild?.tagName !== 'H2').map(sec => sec.id || '(no id)');
      const buttons = [...document.querySelectorAll('button[id]')].map(b => b.id);
      const claims = [...document.querySelectorAll('[data-claim]')].map(e => ({ id: e.dataset.claim, text: e.textContent }));
      const skip = '[data-claim],[data-toy],[data-computed],[data-approx],.readout,.matrix,.deeper,.katex,.tex,code,pre,script,style,nav,[data-pe-player]';
      const walker = document.createTreeWalker(document.querySelector('main') || document.body, NodeFilter.SHOW_TEXT);
      const nums = [];
      for (let n; (n = walker.nextNode());) {
        if (n.parentElement.closest(skip)) continue;
        for (const m of n.textContent.matchAll(/\d[\d,.]*\s?%?/g)) {
          const a = Math.max(0, m.index - 40);
          nums.push(n.textContent.slice(a, m.index + m[0].length + 20).replace(/\s+/g, ' ').trim());
        }
      }
      return { dup, badCh, buttons, claims, nums, chapters: document.querySelectorAll('section.chapter').length };
    });
    if (!s.chapters) fail('no section.chapter found');
    s.dup.forEach(d => fail(`duplicate id: ${d}`));
    s.badCh.forEach(c => fail(`chapter ${c}: needs an id and an <h2> as first child`));
    const btn = new Set(s.buttons);
    const eqs = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.eq[data-build]')]
      .map(e => [e.id, [...e.querySelectorAll('[data-step]')].map(t => +t.dataset.step)])));
    cues.forEach(c => { const m = /^eqstep-(.+)-(\d+)$/.exec(c.id);
      if (m) { if (!eqs[m[1]] || !eqs[m[1]].includes(+m[2])) fail(`${c.f}: cue ${c.id} has no equation #${m[1]} with step ${m[2]}`); }
      else if (!btn.has(c.id)) fail(`${c.f}: cue #${c.id} is not a button on the page`); });
    if (claims) s.claims.forEach(({ id, text }) => {
      const c = claims.get(id);
      if (!c) fail(`data-claim "${id}" missing from claims.json`);
      else if (!norm(text).includes(norm(c.value))) fail(`data-claim "${id}": page says "${text}", claims.json value is "${c.value}"`);
      else if (c.uncertain) fail(`data-claim "${id}" is marked uncertain in claims.json but shown as fact`);
    });
    review = s.nums;
  }

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (overflow > 1) fail(`[${cfg.name}] horizontal overflow of ${overflow}px`);

  const ids = await page.$$eval('main button, figure button', bs =>
    bs.filter(b => !b.closest('[data-pe-player]') && b.offsetParent !== null).map((b, i) => b.id || `#${i}`));
  for (const id of ids) {
    if (id.startsWith('#')) { fail(`[${cfg.name}] button without id (index ${id})`); continue; }
    try { await page.click(`[id="${id}"]`, { timeout: 2000 }); await page.waitForTimeout(50); }
    catch (e) { fail(`[${cfg.name}] clicking #${id}: ${e.message.split('\n')[0]}`); }
  }
  await page.waitForTimeout(200);
  await page.screenshot({ path: path.join(shots, `${cfg.name}.png`), fullPage: true });

  if (narrated && ci === 0) await checkPlayer(page);
  await ctx.close();
}
await browser.close();

async function checkPlayer(page) {
  await page.reload();
  const cueIds = cues.map(c => c.id);
  await page.evaluate(ids => {
    window.__cueClicks = [];
    ids.forEach(id => document.getElementById(id)?.addEventListener('click', () => window.__cueClicks.push(id)));
  }, cueIds);
  const first = await page.$('.pe-ch-play');
  if (!first) return fail('narrated: no chapter play button');
  await first.click();
  await page.waitForTimeout(1500);
  const st = await page.evaluate(() => {
    return { hl: document.querySelectorAll('.pe-s.pe-on').length, warn: document.querySelector('.pe-warn')?.textContent || '' };
  });
  if (st.warn) fail(`narrated: ${st.warn}`);
  const t = await page.$eval('#pe-seek', r => Number(r.value));
  if (!(t > 0.5)) fail(`narrated: audio did not advance after play (t=${t})`);
  if (st.hl !== 1) fail(`narrated: expected 1 highlighted sentence, got ${st.hl}`);
  await page.click('#pe-fwd');
  await page.waitForTimeout(300);
  const [t2, dur] = await page.$eval('#pe-seek', r => [Number(r.value), Number(r.max)]);
  if (!(t2 >= Math.min(t + 9, dur - 0.5))) fail(`narrated: +10s skip did not seek (${t} -> ${t2})`);
  await page.keyboard.press('ArrowLeft');
  await page.waitForTimeout(300);
  const t3 = await page.$eval('#pe-seek', r => Number(r.value));
  if (!(t3 < t2 - 3)) fail(`narrated: ArrowLeft did not seek back (${t2} -> ${t3})`);
  const clicks = await page.evaluate(() => window.__cueClicks);
  console.log(`narrated: cue clicks observed: ${clicks.join(', ') || '(none)'}`);
}

if (review.length) {
  console.log(`\nREVIEW ${review.length} bare number(s) in prose (should each be data-claim, data-toy, or plainly not a result):`);
  review.forEach(r => console.log(`  ... ${r}`));
}
if (fails.length) {
  console.log(`\nFAIL (${fails.length}):`);
  fails.forEach(f => console.log(`  - ${f}`));
  process.exit(1);
}
console.log(`\nPASS  screenshots: ${shots}`);

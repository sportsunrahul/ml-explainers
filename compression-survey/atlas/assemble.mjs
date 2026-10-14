// node assemble.mjs [outDir] [id ...]   (any cwd)
// Joins atlas/parts/<id>.{article.html,scene.js,dialogs.html} in atlas/ORDER into the six files that the skill's
// atlas build.mjs expects, under <outDir>/atlas/, then builds <outDir>/atlas.html. Default outDir: the project.
// With ids, only those chapters are built (a builder's private test). Articles use {{NUM}} and {{CONTINUE}}.
import fs from 'fs'; import path from 'path'; import { execFileSync } from 'child_process'; import { fileURLToPath } from 'url';
const here = path.dirname(fileURLToPath(import.meta.url)), proj = path.dirname(here);
const out = path.resolve(process.argv[2] || proj), only = process.argv.slice(3);
const read = f => fs.readFileSync(path.join(here, f), 'utf8'), has = f => fs.existsSync(path.join(here, f));
let order = read('ORDER').trim().split('\n').map(l => { const [id, name] = l.split('|'); return { id, name }; });
order = order.filter(o => has(`parts/${o.id}.article.html`) && (!only.length || only.includes(o.id)));
if (!order.length) throw new Error('no chapters to build');
const gen = path.join(out === proj ? path.join(proj, 'build', 'atlas-gen') : out, 'atlas'); fs.mkdirSync(gen, { recursive: true });
const meta = JSON.parse(read('meta.json')); meta.out = path.join(out, 'atlas.html');
const chips = order.map(o => `      <button type="button" data-go="${o.id}">${o.name}</button>`).join('\n');
const chapters = order.map((o, i) => {
  const nx = order[i + 1];
  const cont = nx ? `<button type="button" class="continue" data-go="${nx.id}">Continue with ${nx.name}</button>`
    : '<div class="end-links"><a class="btn" href="explainer.html">Read the full explainer</a><button class="btn" type="button" data-go="title">Back to the title</button></div>';
  return read(`parts/${o.id}.article.html`).replace('{{NUM}}', `${i + 1} of ${order.length}`).replace('{{CONTINUE}}', cont);
}).join('\n');
const seen = new Set(), dlg = [];
for (const f of ['dialogs-common.html', ...order.map(o => `parts/${o.id}.dialogs.html`)]) {
  if (!has(f)) continue;
  for (const m of read(f).matchAll(/<dialog id="([^"]+)"[\s\S]*?<\/dialog>/g)) if (!seen.has(m[1])) { seen.add(m[1]); dlg.push(m[0]); }
}
const kits = fs.readdirSync(path.join(proj, 'parts', 'kit')).filter(f => f.endsWith('.js')).sort().map(f => fs.readFileSync(path.join(proj, 'parts', 'kit', f), 'utf8')).join('\n');
const scenes = kits + '\n' + order.map(o => has(`parts/${o.id}.scene.js`) ? read(`parts/${o.id}.scene.js`) : '').join('\n');
const W = (f, s) => fs.writeFileSync(path.join(gen, f), s);
W('meta.json', JSON.stringify(meta, null, 1)); W('hero.html', read('hero.html')); W('chips.html', chips);
W('chapters.html', chapters); W('dialogs.html', dlg.join('\n')); W('scenes.js', scenes);
execFileSync('node', [path.join(process.env.HOME, '.claude/skills/paper-explainer/assets/atlas/build.mjs'), path.dirname(gen)], { stdio: 'inherit' });

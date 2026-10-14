# Build the long page and the atlas.
#   python3 build/build.py                 -> parts/ORDER into explainer.src.html / explainer.html, then check-page
#   python3 build/build.py ORDERFILE OUTDIR -> a private build (e.g. one chapter) into OUTDIR
#   python3 build/build.py --atlas         -> atlas/scenes.js from the shared parts, then the skill's atlas build
import base64, glob, json, os, subprocess, sys

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.expanduser('~/.claude/skills/paper-explainer/assets')
NODE = os.path.expanduser('~/paper-explainers/.node')


def merge_claims(out_dir):
    allc = []
    for f in sorted(glob.glob(f'{P}/notes/claims/*.json')):
        allc += [dict(c, _file=os.path.basename(f)) for c in json.load(open(f))]
    by = {}
    for c in allc:
        if c['id'] in by and by[c['id']]['value'] != c['value']:
            sys.exit(f"conflicting claim values for {c['id']}: {by[c['id']]['_file']} vs {c['_file']}")
        by.setdefault(c['id'], c)
    os.makedirs(f'{out_dir}/notes', exist_ok=True)
    json.dump([{k: v for k, v in c.items() if k != '_file'} for c in by.values()], open(f'{out_dir}/notes/claims.json', 'w'), indent=1)
    print(len(by), 'claims merged')


def data_js():
    D = {}
    for name in ['runA', 'runB', 'runC', 'poolsim', 'scan']:
        f = f'{P}/build/data/{name}.json'
        if os.path.exists(f):
            D[name] = json.load(open(f))
    S = {}
    for name in ['hero', 'tiles2']:
        w = f'{P}/build/data/{name}.webp'
        if os.path.exists(w):
            S[name] = {'src': 'data:image/webp;base64,' + base64.b64encode(open(w, 'rb').read()).decode(),
                       **json.load(open(f'{P}/build/data/{name}.json'))}
    return 'const DATA = ' + json.dumps(D, separators=(',', ':')) + ';\nconst SPRITES = ' + json.dumps(S, separators=(',', ':')) + ';\n'


def part(name):
    if name == 'gen:data':
        return '<script>\n' + data_js() + '</script>\n'
    s = open(f'{P}/parts/{name}').read()
    if name.endswith('.js') and not s.lstrip().startswith('<'):
        s = '<script>\n' + s + '\n</script>\n'
    return s


def long_page(order, out):
    os.makedirs(out, exist_ok=True)
    merge_claims(out)
    names = [l.strip() for l in open(order) if l.strip() and not l.startswith('#')]
    html = open(f'{P}/explainer.head.part').read() + open(f'{P}/parts/body-top.html').read()
    html += ''.join(part(n) for n in names) + '</main>\n<script>KIT.legends();</script>\n</body>\n</html>\n'
    bad = [i for i, ch in enumerate(html) if ord(ch) > 127]
    if bad:
        i = bad[0]
        print('WARNING: non-ASCII character near:', repr(html[max(0, i - 40):i + 10]))
    open(f'{out}/explainer.src.html', 'w').write(html)
    subprocess.run(['node', f'{SKILL}/prerender-math.mjs', f'{out}/explainer.src.html', f'{out}/explainer.html'], cwd=NODE, check=True)
    r = subprocess.run(['node', f'{SKILL}/check-page.mjs', out], cwd=NODE)
    print('check-page exit', r.returncode)


def atlas():
    # Each stage owns atlas/parts/NN-<id>.html, atlas/glue/NN-<id>.js and optional atlas/dialogs/<id>.html.
    A = f'{P}/atlas'
    open(f'{A}/chapters.html', 'w').write(''.join(open(f).read() for f in sorted(glob.glob(f'{A}/parts/*.html'))))
    open(f'{A}/dialogs.html', 'w').write(''.join(open(f).read() for f in sorted(glob.glob(f'{A}/dialogs/*.html'))))
    libs = ['kit.js', 'toy-pool.js'] + sorted(os.path.basename(f) for f in glob.glob(f'{P}/parts/fig-*.js'))
    js = data_js() + '\n'.join(open(f'{P}/parts/{n}').read() for n in libs) + '\n' + '\n'.join(open(f).read() for f in sorted(glob.glob(f'{A}/glue/*.js')))
    open(f'{A}/scenes.js', 'w').write(js)
    subprocess.run(['node', f'{SKILL}/atlas/build.mjs', P], cwd=NODE, check=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--atlas':
        atlas()
    else:
        order = sys.argv[1] if len(sys.argv) > 1 else f'{P}/parts/ORDER'
        out = sys.argv[2] if len(sys.argv) > 2 else P
        long_page(order if os.path.isabs(order) else os.path.join(P, order), out if os.path.isabs(out) else os.path.join(P, out))

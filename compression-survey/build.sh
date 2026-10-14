# Usage: bash build.sh [ORDERFILE] [OUTDIR] [--no-check]
# Assembles explainer.src.html from parts/ (head, body-top, libs, kits, chapters in ORDER), merges claims,
# prerenders math into explainer.html, then runs the skill's page check.
# OUTDIR (default: the project root) lets parallel builders test privately: the page, notes/claims.json and
# build/shots all land under OUTDIR, e.g. build/test-distill.
set -e
P=~/paper-explainers/compression-survey
ORDER=${1:-$P/parts/ORDER}; OUT=${2:-$P}
case "$ORDER" in /*) ;; *) ORDER="$P/$ORDER";; esac
case "$OUT" in /*) ;; *) OUT="$P/$OUT";; esac
mkdir -p "$OUT/notes"
python3 -I - "$P" "$OUT" <<'EOF'
import json, glob, sys
P, OUT = sys.argv[1], sys.argv[2]; a = []
for f in sorted(glob.glob(P + '/notes/claims/*.json')): a += json.load(open(f))
ids = [c['id'] for c in a]; dup = sorted(set(i for i in ids if ids.count(i) > 1))
conf = [i for i in dup if len(set(c['value'] for c in a if c['id'] == i)) > 1]
if conf: sys.exit('conflicting claim values for ids: %s' % conf)
seen = set(); a = [c for c in a if not (c['id'] in seen or seen.add(c['id']))]
json.dump(a, open(OUT + '/notes/claims.json', 'w'), indent=1)
print(len(a), 'claims merged')
EOF
# The Write tool turns \u escapes back into characters; re-escape kits (and atlas scenes) instead of failing.
python3 -I -c "
import glob,sys
for f in glob.glob(sys.argv[1]+'/parts/kit/*.js')+glob.glob(sys.argv[1]+'/atlas/parts/*.js'):
    s=open(f,encoding='utf-8').read(); t=''.join(c if ord(c)<128 else '\\\\u%04x'%ord(c) for c in s).replace('\t','  ')
    if t!=s: open(f,'w').write(t); print('escaped',f)
" "$P"
{ cat $P/parts/head.html $P/parts/body-top.html $P/parts/lib-page-chrome.html $P/parts/lib-pe3.html $P/parts/lib-archflow.html
  printf '<script>\n'; cat $P/parts/kit/*.js; printf '</script>\n'
  cat $P/parts/hero-mount.html
  for p in $(cat "$ORDER"); do cat $P/parts/$p; done
  printf '</main>\n</body>\n</html>\n'; } > "$OUT/explainer.src.html"
cd ~/paper-explainers/.node
node ~/.claude/skills/paper-explainer/assets/prerender-math.mjs "$OUT/explainer.src.html" "$OUT/explainer.html"
[ "$3" = "--no-check" ] || [ "$2" = "--no-check" ] || node ~/.claude/skills/paper-explainer/assets/check-page.mjs "$OUT"

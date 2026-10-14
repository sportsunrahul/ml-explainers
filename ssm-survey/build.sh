# Usage: bash build.sh [ORDERFILE] [OUTDIR]
# Default: parts/ORDER -> project root. core-*.js files are shared with the atlas: wrapped in <script> here,
# concatenated raw (before atlas/scenes.src.js) into atlas/scenes.js for the atlas build.
set -e
P=~/paper-explainers/ssm-survey
ORDER=${1:-$P/parts/ORDER}; OUT=${2:-$P}
case "$ORDER" in /*) ;; *) ORDER="$P/$ORDER";; esac
case "$OUT" in /*) ;; *) OUT="$P/$OUT";; esac
mkdir -p "$OUT/notes"
python3 -I -c "
import json,glob,sys
a=[]
for f in sorted(glob.glob('$P/notes/claims/*.json')): a+=json.load(open(f))
ids=[c['id'] for c in a]; d=sorted(set(i for i in ids if ids.count(i)>1))
conf=[i for i in d if len(set(c['value'] for c in a if c['id']==i))>1]
if conf: sys.exit('conflicting claim values for ids: %s'%conf)
seen=set(); a=[c for c in a if not (c['id'] in seen or seen.add(c['id']))]
json.dump(a,open('$OUT/notes/claims.json','w'),indent=1)
print(len(a),'claims merged')"
{ cat $P/explainer.head.part $P/parts/body-top.html
  for p in $(cat "$ORDER"); do
    case "$p" in core-*.js) printf '<script>\n'; cat $P/parts/$p; printf '</script>\n';; *) cat $P/parts/$p;; esac
  done
  printf '</main>\n</body>\n</html>\n'; } > "$OUT/explainer.src.html"
cd ~/paper-explainers/.node
node ~/.claude/skills/paper-explainer/assets/prerender-math.mjs "$OUT/explainer.src.html" "$OUT/explainer.html"
[ "$NOCHECK" = 1 ] || node ~/.claude/skills/paper-explainer/assets/check-page.mjs "$OUT"

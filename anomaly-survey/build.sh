# Usage: bash build.sh [ORDERFILE] [OUTDIR]
# Default: parts/ORDER -> project root. With OUTDIR (a private test build), claims are merged into OUTDIR/notes.
set -e
P=~/paper-explainers/anomaly-survey
ORDER=${1:-$P/parts/ORDER}; OUT=${2:-$P}
case "$ORDER" in /*) ;; *) ORDER="$P/$ORDER";; esac
case "$OUT" in /*) ;; *) OUT="$P/$OUT";; esac
mkdir -p "$OUT/notes"
python3 -I -c "
import json,glob,sys
a=[]
for f in sorted(glob.glob('$P/notes/claims/*.json')): a+=json.load(open(f))
ids=[c['id'] for c in a]; d=sorted(set(i for i in ids if ids.count(i)>1))
# the same id in two files is fine when the values agree (dates extracted by two agents); a conflict is an error
conf=[i for i in d if len(set(c['value'] for c in a if c['id']==i))>1]
if conf: sys.exit('conflicting claim values for ids: %s'%conf)
d=[]
if d: print('WARNING (private build, keeping first occurrence): duplicate claim ids', d)
seen=set(); a=[c for c in a if not (c['id'] in seen or seen.add(c['id']))]
json.dump(a,open('$OUT/notes/claims.json','w'),indent=1)
print(len(a),'claims merged')"
{ cat $P/explainer.head.part $P/parts/body-top.html
  for p in $(cat "$ORDER"); do cat $P/parts/$p; done
  printf '</main>\n</body>\n</html>\n'; } > "$OUT/explainer.src.html"
cd ~/paper-explainers/.node
node ~/.claude/skills/paper-explainer/assets/prerender-math.mjs "$OUT/explainer.src.html" "$OUT/explainer.html"
node ~/.claude/skills/paper-explainer/assets/check-page.mjs "$OUT"

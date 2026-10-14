# Usage: bash build-atlas.sh [ATLAS_ORDER] [OUTDIR] [CORE_ORDER]
# Assembles atlas/{chips,chapters,dialogs}.html and atlas/scenes.js from atlas/parts/<slug>.{chip,article,dialogs}.html and
# <slug>.scene.js in ATLAS_ORDER (default atlas/ORDER), with the core-*.js files listed in CORE_ORDER (default parts/ORDER)
# prepended to the scenes. Default OUTDIR is the project (writes ssm-atlas.html); a private OUTDIR gets its own copy.
# Dialogs are deduplicated by id (first one wins; later duplicates are reported).
set -e
P=~/paper-explainers/ssm-survey
AO=${1:-$P/atlas/ORDER}; OUT=${2:-$P}; CO=${3:-$P/parts/ORDER}
for v in AO OUT CO; do eval "x=\$$v"; case "$x" in /*) ;; *) eval "$v=\$P/\$x";; esac; done
A=$OUT/atlas; mkdir -p $A
[ "$OUT" = "$P" ] || { cp $P/atlas/hero.html $A/; python3 -I -c "import json;m=json.load(open('$P/atlas/meta.json'));m['out']='../ssm-atlas.html';json.dump(m,open('$A/meta.json','w'))"; }
python3 -I - "$P" "$AO" "$CO" "$A" <<'PY'
import sys, re, os
P, AO, CO, A = sys.argv[1:]
slugs = [l.strip() for l in open(AO) if l.strip() and not l.startswith('#')]
cores = [l.strip() for l in open(CO) if l.strip().startswith('core-')]
part = lambda s, ext: open(f'{P}/atlas/parts/{s}.{ext}').read() if os.path.exists(f'{P}/atlas/parts/{s}.{ext}') else ''
open(f'{A}/chips.html', 'w').write(''.join(part(s, 'chip.html') for s in slugs))
open(f'{A}/chapters.html', 'w').write('\n'.join(part(s, 'article.html') for s in slugs))
seen, out = set(), []
for s in slugs:
    for m in re.finditer(r'<dialog id="([^"]+)".*?</dialog>', part(s, 'dialogs.html'), re.S):
        if m.group(1) in seen: print('duplicate dialog id kept first:', m.group(1), 'from', s); continue
        seen.add(m.group(1)); out.append(m.group(0))
open(f'{A}/dialogs.html', 'w').write('\n'.join(out) + '\n')
sc = ''.join(open(f'{P}/parts/{c}').read() + '\n' for c in cores) + part('_head', 'scene.js') + '\n'.join(part(s, 'scene.js') for s in slugs)
open(f'{A}/scenes.js', 'w').write(sc)
print(len(slugs), 'chapters,', len(cores), 'cores,', len(out), 'dialogs')
PY
cd ~/paper-explainers/.node
node ~/.claude/skills/paper-explainer/assets/atlas/build.mjs $OUT
[ "$NOCHECK" = 1 ] || node ~/.claude/skills/paper-explainer/assets/atlas/test.mjs $OUT/ssm-atlas.html $OUT/build/atlas-shots

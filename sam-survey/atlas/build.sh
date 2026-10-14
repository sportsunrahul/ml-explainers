# Assemble atlas/scenes.js from the shared parts, then fill the skill's atlas shell.
set -e
P=~/paper-explainers/sam-survey
{ cat $P/parts/lib-toy.js; for s in $(cat $P/atlas/SCENES); do cat $P/parts/scene-$s.js; done
  printf '%s\n' "/* Atlas: 2D scenes keep the bottom 130 px free on desktop, where the readout box sits. */" \
    "SCENES.inspection = SCENES.inspect;" \
    "for (const k of Object.keys(SCENES)) { const f = SCENES[k]; SCENES[k] = ctx => f(Object.assign(ctx, { reserve: matchMedia('(min-width: 861px)').matches ? 130 : 0 })); }"; } > $P/atlas/scenes.js
cd ~/paper-explainers/.node && node ~/.claude/skills/paper-explainer/assets/atlas/build.mjs $P

#!/bin/bash
# Assemble explainer.src.html from the parts, then prerender explainer.html.
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P/build"
cat head.html top.html lib-page-chrome.js hero.html part1.html training.html part2.html part3.html > "$P/explainer.src.html"
cd ~/paper-explainers/.node
node ~/.claude/skills/paper-explainer/assets/prerender-math.mjs "$P/explainer.src.html" "$P/explainer.html"

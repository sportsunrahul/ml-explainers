# Build issues found by the continuous/mamba agent

- atlas/build.mjs rejects non-ASCII in scenes.js. Seen on 2026-10-10: parts/core-s4.js (lines 103, 110, 116), parts/core-scan.js (170, 174, 216) and parts/core-vision.js (133) have literal Greek letters, bars or superscripts in strings. Once those cores are in the shared ORDER the atlas build will stop. Use \u escapes.
- LSSL App B.2 plugs HiPPO's positive A into x' = Ax + Bu without the minus (already in notes); this page avoids it by using a hand-picked stable A in ch-continuous.

"""Mechanical design-pass edits shared by the six "explained in 3D" pages.

Usage from a page script:
    import sys; sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
    import common
    s = common.apply(s, hl='var(--acc)')
Every replacement asserts its anchor matches exactly once, so a template drift fails loudly.
"""
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KATEX_LINK = ('<link id="katex-css" rel="stylesheet" '
              'href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.18.9/katex.min.css" '
              'integrity="sha384-lPx0C4zIUZLpveABMwOFcFeGZwsvKBJfhJ85FN1PYOV7xApBcFMhcAEMVKF8loOI" '
              'crossorigin="anonymous">')

STAGE_OLD = "const s={fig,wrap,rotY:0,rotX:0,tRotY:0,tRotX:0,auto:0};"
# s.g(word) is 0..1 for 1.1 s after an equation term with that link word fires; build() code reads it in update().
STAGE_NEW = (STAGE_OLD + "\n s.glowT={};fig.addEventListener('pe-link',e=>{s.glowT[e.detail]=performance.now();});"
             "s.g=k=>{const t0=s.glowT[k];if(!t0)return 0;const u=(performance.now()-t0)/1100;return u>=1?0:reduced?1:Math.sin(Math.PI*u);};")


def once(s, old, new):
    n = s.count(old)
    assert n == 1, f'expected 1 match, found {n}: {old[:80]!r}'
    return s.replace(old, new)


def split_middots(s):
    """Footer source lists join several references with a middle dot; give each its own list item."""
    a, b = s.index('<footer'), s.index('</footer>')
    foot = s[a:b]
    foot = re.sub(r'<li>(.*?)</li>', lambda m: ''.join(f'<li>{p.strip()}</li>' for p in m.group(1).split(' · ')), foot, flags=re.S)
    return s[:a] + foot + s[b:]


def apply(s, hl, extra_css=''):
    css = open(os.path.join(HERE, 'chrome.css')).read()
    js = open(os.path.join(HERE, 'chrome.js')).read()
    s = once(s, '</style>', f':root{{--hl:{hl}}}\n'
             # A div that is both .wrap and .prose was centered at 700px while other sections sat at the wrap's left edge.
             '.wrap.prose{max-width:1120px}.wrap.prose>*{max-width:700px}\n'
             + css + extra_css + '</style>')
    s = once(s, '</head>', KATEX_LINK + '\n</head>')
    s = once(s, STAGE_OLD, STAGE_NEW)
    s = once(s, '</body>', '<script>\n' + js + '</script>\n</body>')
    s = split_middots(s)
    return s

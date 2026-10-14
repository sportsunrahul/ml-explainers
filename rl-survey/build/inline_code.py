"""Replace <!--code:PATH--> and <!--out:PATH--> in the assembled page with the exercise file and its recorded output.

PATH is relative to code/, e.g. ch02-bandits/numpy_ex.py or kit/cartpole.py. The output of chNN/x.py is
code/out/chNN--x.py.txt, written by code/run_all.py. Missing files or non-ASCII text stop the build.
"""
import html
import io
import keyword
import re
import sys
import tokenize
from pathlib import Path

CODE, PAGE = Path(sys.argv[1]), Path(sys.argv[2])


def read_ascii(path):
    text = path.read_text()
    bad = [c for c in text if ord(c) > 127]
    assert not bad, f'non-ASCII {bad[:5]} in {path}'
    return text.rstrip('\n')


def highlight(src):
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    out, pos = [], 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        start = offsets[tok.start[0] - 1] + tok.start[1]
        end = offsets[tok.end[0] - 1] + tok.end[1]
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.NAME and keyword.iskeyword(tok.string)):
            cls = 'c' if tok.type == tokenize.COMMENT else 'k'
            out.append(html.escape(src[pos:start], quote=False))
            out.append(f'<span class="{cls}">{html.escape(src[start:end], quote=False)}</span>')
            pos = end
    out.append(html.escape(src[pos:], quote=False))
    return ''.join(out)


def code_block(m):
    return '<pre class="code"><code>' + highlight(read_ascii(CODE / m.group(1))) + '</code></pre>'


def out_block(m):
    folder, script = m.group(1).split('/')
    text = read_ascii(CODE / 'out' / f'{folder}--{script}.txt')
    return '<pre class="code-out"><code>' + html.escape(text, quote=False) + '</code></pre>'


page = PAGE.read_text()
page, n_code = re.subn(r'<!--code:([\w./-]+)-->', code_block, page)
page, n_out = re.subn(r'<!--out:([\w./-]+)-->', out_block, page)
PAGE.write_text(page)
print(f'inlined {n_code} code blocks and {n_out} outputs')

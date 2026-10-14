"""Build job/sentences.json from scripts/NN-*.md (one sentence per line, blank line = paragraph break) and lexicon.json."""
import re, json, glob, os
P = os.path.dirname(os.path.abspath(__file__)) + "/.."
lex = json.load(open(f"{P}/lexicon.json"))
keys = sorted(lex, key=len, reverse=True)
def speak(t):
    for k in keys:
        say = lex[k].get("qwen3", lex[k]["say"])
        t = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", say, t)
    return t
out = []
for f in sorted(glob.glob(f"{P}/scripts/*.md")):
    txt = open(f).read(); fm = dict(l.split(": ", 1) for l in txt.split("---")[1].strip().splitlines())
    n = int(os.path.basename(f)[:2]); idx = 0; para = 0
    for line in txt.split("---", 2)[2].strip("\n").split("\n"):
        if not line.strip(): para += 1; continue
        text = re.sub(r"\s*\[\[cue:[\w-]+\]\]", "", line).strip()
        out.append(dict(chapter=fm["chapter"], n=n, idx=idx, para=para, text=text, tts=speak(text), cues=re.findall(r"\[\[cue:([\w-]+)\]\]", line)))
        idx += 1
json.dump(out, open(f"{P}/job/sentences.json", "w"), indent=1)
print(len(out), "sentences;", sum(len(s['text'].split()) for s in out), "words")

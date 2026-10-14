import json, sys
m = json.load(open("out/manifest.json"))
v = sys.argv[1] if len(sys.argv) > 1 else "A"
d = m["voices"][v]
S = d["sentences"]
lines = [f"# Audio QA report (voice {v})", "",
         f"Model revisions: {m['revisions']}. Median rate {d['median_wps']:.2f} words/s; speaker-similarity floor {d['sim_floor']:.2f}.", ""]
retried = [s for s in S if len(s["takes"]) > 1]
flagged = [s for s in S if s["flags"]]
lines += [f"- Sentences: {len(S)}; retried: {len(retried)}; still flagged after up to 3 retries: {len(flagged)}.",
          f"- Mean WER of chosen takes: {sum(next(t for t in s['takes'] if t['seed'] == s['chosen_seed'])['wer'] for s in S) / len(S):.3f}.", ""]
lines += ["## Still flagged (listen to these)", ""]
if not flagged:
    lines.append("None.")
for s in flagged:
    t = next(t for t in s["takes"] if t["seed"] == s["chosen_seed"])
    lines += [f"- **{s['chapter']} #{s['idx']}**: {', '.join(s['flags'])}",
              f"  - script: {s['text']}", f"  - heard:  {t['hyp']}", f"  - file: out/{v}/wav/{s['chapter']}-{s['idx']}.wav"]
lines += ["", "## Retried and fixed", ""]
for s in retried:
    if not s["flags"]:
        lines.append(f"- {s['chapter']} #{s['idx']}: first take flagged ({', '.join(s['takes'][0]['flags'])}); fixed on seed {s['chosen_seed']}.")
open("qa_report.md", "w").write("\n".join(lines) + "\n")
print(f"{len(S)} sentences, {len(retried)} retried, {len(flagged)} flagged")

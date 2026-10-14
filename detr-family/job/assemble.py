import re, json, subprocess, glob, os, sys, numpy as np, soundfile as sf
# usage: assemble.py <wav_dir> <out_dir> [chapter-glob]
WAV, OUT = sys.argv[1], sys.argv[2]; GLOB = sys.argv[3] if len(sys.argv) > 3 else "*"
os.makedirs(OUT, exist_ok=True)
SR = 24000
TRIM = ("silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.02,areverse,"
        "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.02,areverse")
LN = "loudnorm=I=-16:TP=-1.5:LRA=11"
def ff(*a): return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", *a],
                                  capture_output=True, text=True, check=True)
def ln_json(stderr): return json.loads(re.findall(r"\{[^{}]*\}", stderr)[-1])  # trailing log lines follow the JSON

def parse(path):
    txt = open(path).read()
    fm = dict(l.split(": ", 1) for l in txt.split("---")[1].strip().splitlines())
    sents, para = [], 0
    for line in txt.split("---", 2)[2].strip("\n").split("\n"):
        if not line.strip(): para += 1; continue
        sents.append(dict(text=re.sub(r"\s*\[\[cue:[\w-]+\]\]", "", line).strip(),
                          cues=re.findall(r"\[\[cue:([\w-]+)\]\]", line), para=para))
    return fm, sents

for f in sorted(glob.glob(f"scripts/{GLOB}.md")):
    nn = os.path.basename(f)[:2]; fm, sents = parse(f)
    pieces, t, prev = [np.zeros(int(.3 * SR), np.float32)], .3, None
    for i, s in enumerate(sents):
        src = f"{WAV}/{fm['chapter']}-{i}.wav"; tw = f"{OUT}/trim/{nn}-{i}.wav"
        os.makedirs(f"{OUT}/trim", exist_ok=True)
        ff("-i", src, "-af", TRIM, "-ac", "1", "-ar", str(SR), tw)
        if prev is not None:
            gap = int((.7 if s["para"] != prev else .25) * SR)
            pieces.append(np.zeros(gap, np.float32)); t += gap / SR
        a, _ = sf.read(tw, dtype="float32")
        s["t0"] = round(t, 3); t += len(a) / SR; s["t1"] = round(t, 3)
        pieces.append(a); prev = s["para"]
    pieces.append(np.zeros(int(.3 * SR), np.float32))
    sf.write(f"{OUT}/{nn}.wav", np.concatenate(pieces), SR)
    m = ln_json(ff("-i", f"{OUT}/{nn}.wav", "-af", f"{LN}:print_format=json", "-f", "null", "-").stderr)
    ff("-i", f"{OUT}/{nn}.wav", "-af", f"{LN}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
       f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
       f":offset={m['target_offset']}:linear=true", "-ar", str(SR), f"{OUT}/{nn}.norm.wav")
    d0, d1 = sf.info(f"{OUT}/{nn}.wav").duration, sf.info(f"{OUT}/{nn}.norm.wav").duration
    assert abs(d0 - d1) < .02, (nn, d0, d1)
    ff("-i", f"{OUT}/{nn}.norm.wav", "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", f"{OUT}/{nn}.mp3")
    json.dump(dict(id=fm["chapter"], title=fm["title"], duration=round(d1, 3), sentences=sents),
              open(f"{OUT}/{nn}.timings.json", "w"), indent=1)

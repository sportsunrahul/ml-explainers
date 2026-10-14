import os, sys, json, time, hashlib, statistics
import numpy as np, soundfile as sf, torch, torchaudio, jiwer
from huggingface_hub import snapshot_download

P = os.path.dirname(os.path.abspath(__file__)) + "/.."
cfg = json.load(open(f"{P}/job/config.json"))
S = json.load(open(f"{P}/job/sentences.json"))
CACHE = f"{P}/cache"; os.makedirs(CACHE, exist_ok=True)
PROG = f"{P}/job/progress.json"
lex = json.load(open(f"{P}/lexicon.json"))
GLOSSARY = ", ".join(k for k in lex)
VOICES = cfg["voices"]  # name -> {kind: design|custom, instruct, speaker?}
todo = [(v, s) for v in cfg["run_voices"] for s in S]
done = 0

def progress(**kw):
    json.dump({"done": done, "total": len(todo), **kw}, open(PROG, "w"))

def rev(mid):
    path = snapshot_download(mid); return path, os.path.basename(path)

def load(mid):
    from qwen_tts import Qwen3TTSModel
    path, r = rev(mid)
    return Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"), r

def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# 1. Reference clips for designed voices (VoiceDesign model, loaded only if needed).
REF_TEXT = cfg["ref_text"]
os.makedirs(f"{P}/voice", exist_ok=True)
need = [v for v in cfg["run_voices"] if VOICES[v]["kind"] == "design" and not os.path.exists(f"{P}/voice/{v}.ref.wav")]
if need:
    vd, _ = load("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    for v in need:
        seed_all(cfg["base_seed"])
        wavs, sr = vd.generate_voice_design(text=REF_TEXT, language="English", instruct=VOICES[v]["instruct"])
        sf.write(f"{P}/voice/{v}.ref.wav", np.asarray(wavs[0], dtype=np.float32), sr)
    del vd; torch.cuda.empty_cache()

# 2. Load synthesis, ASR and speaker models together (~12 GB on a 24 GB card).
base = cv = None; revs = {}
if any(VOICES[v]["kind"] == "design" for v in cfg["run_voices"]):
    base, revs["base"] = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
if any(VOICES[v]["kind"] == "custom" for v in cfg["run_voices"]):
    cv, revs["custom"] = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
prompts = {}
for v in cfg["run_voices"]:
    if VOICES[v]["kind"] == "design":
        pp = f"{P}/voice/{v}.prompt.pt"
        if os.path.exists(pp):
            prompts[v] = torch.load(pp, weights_only=False)
        else:
            prompts[v] = base.create_voice_clone_prompt(ref_audio=f"{P}/voice/{v}.ref.wav", ref_text=REF_TEXT)
            torch.save(prompts[v], pp)

from faster_whisper import WhisperModel
from whisper_normalizer.english import EnglishTextNormalizer
from speechbrain.inference.speaker import EncoderClassifier
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
norm = EnglishTextNormalizer()
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.environ["HF_HOME"] + "/ecapa", run_opts={"device": "cuda"})

def to16(w, sr):
    return torchaudio.functional.resample(torch.from_numpy(w), sr, 16000)

def emb(w, sr):
    return enc.encode_batch(to16(w, sr)[None].cuda()).squeeze()

ref_emb = {}
for v in cfg["run_voices"]:
    if VOICES[v]["kind"] == "design":
        w, sr = sf.read(f"{P}/voice/{v}.ref.wav", dtype="float32"); ref_emb[v] = emb(w, sr)

def voice_id(v):
    c = VOICES[v]
    if c["kind"] == "design":
        return hashlib.sha256(open(f"{P}/voice/{v}.ref.wav", "rb").read() + REF_TEXT.encode()).hexdigest()[:16]
    return f"custom:{c['speaker']}:{c.get('instruct', '')}"

def key(v, s, seed, sig=None):
    # sig: hash of the batch's texts. Batched output depends on the whole batch, so it enters the
    # key; batch size 1 leaves it out, keeping keys of takes made before batching existed.
    r = revs["base"] if VOICES[v]["kind"] == "design" else revs["custom"]
    return hashlib.sha256(json.dumps([s["tts"], "qwen3", r, voice_id(v), cfg["sampling"], seed] + ([sig] if sig else []), sort_keys=True).encode()).hexdigest()[:20]

B = int(cfg.get("batch", 1))

def synth_many(v, items):
    """items: [(s, seed)]; returns [(w, sr, path, gen_s)] in order. Uncached items are generated in
    batches of B (one seed per batch: the first item's)."""
    res = [None] * len(items)
    todo_i = []
    for n, (s, seed) in enumerate(items):
        path = f"{CACHE}/{key(v, s, seed)}.wav"
        if os.path.exists(path):  # reuse any existing take, including pre-batching ones
            w, sr = sf.read(path, dtype="float32"); res[n] = (w, sr, path, 0.0)
        else:
            todo_i.append(n)
    for c in range(0, len(todo_i), B):
        chunk = todo_i[c:c + B]
        if len(chunk) == 1 and B == 1:
            s, seed = items[chunk[0]]; res[chunk[0]] = synth(v, s, seed); continue
        texts = [items[n][0]["tts"] for n in chunk]; seed = items[chunk[0]][1]
        sig = hashlib.sha256("\x1f".join(texts).encode()).hexdigest()[:12]
        paths = [f"{CACHE}/{key(v, items[n][0], seed, sig)}.wav" for n in chunk]
        if all(os.path.exists(pp) for pp in paths):
            for n, pp in zip(chunk, paths):
                w, sr = sf.read(pp, dtype="float32"); res[n] = (w, sr, pp, 0.0)
            continue
        seed_all(seed); t = time.time()
        if VOICES[v]["kind"] == "design":
            wavs, sr = base.generate_voice_clone(text=texts, language=["English"] * len(texts), voice_clone_prompt=prompts[v] * len(texts))
        else:
            wavs, sr = cv.generate_custom_voice(text=texts, language=["English"] * len(texts), speaker=[VOICES[v]["speaker"]] * len(texts), instruct=[VOICES[v].get("instruct")] * len(texts))
        g = (time.time() - t) / len(texts)
        for n, pp, w in zip(chunk, paths, wavs):
            w = np.asarray(w, dtype=np.float32); sf.write(pp, w, sr); res[n] = (w, sr, pp, g)
    return res

def synth(v, s, seed):
    k = key(v, s, seed); path = f"{CACHE}/{k}.wav"
    if os.path.exists(path):
        w, sr = sf.read(path, dtype="float32"); return w, sr, path, 0.0
    seed_all(seed); t = time.time()
    if VOICES[v]["kind"] == "design":
        wavs, sr = base.generate_voice_clone(text=s["tts"], language="English", voice_clone_prompt=prompts[v])
    else:
        wavs, sr = cv.generate_custom_voice(text=s["tts"], language="English", speaker=VOICES[v]["speaker"], instruct=VOICES[v].get("instruct"))
    w = np.asarray(wavs[0], dtype=np.float32); sf.write(path, w, sr)
    return w, sr, path, time.time() - t

def trimmed(w, sr, db=-45):
    f = int(.02 * sr); n = len(w) // f
    rms = np.array([np.sqrt(np.mean(w[i * f:(i + 1) * f] ** 2) + 1e-12) for i in range(n)])
    loud = 20 * np.log10(rms) > db
    if not loud.any(): return w, 0.0
    a, b = np.argmax(loud), n - np.argmax(loud[::-1])
    gaps, run = [], 0
    for x in loud[a:b]:
        run = 0 if x else run + 1; gaps.append(run)
    return w[a * f:b * f], max(gaps) * .02

def measure(v, s, w, sr):
    segs, _ = asr.transcribe(to16(w, sr).numpy(), language="en", initial_prompt=GLOSSARY, beam_size=5)
    hyp = " ".join(x.text for x in segs).strip()
    dh = lambda x: norm(x.replace("-", ""))  # "cleanup" vs "clean-up" is not an error
    m = jiwer.process_words(dh(s["text"]), dh(hyp))
    core, gap = trimmed(w, sr)
    words = len(s["text"].split()); dur = len(core) / sr
    sim = float(torch.nn.functional.cosine_similarity(ref_emb[v], emb(w, sr), dim=0)) if v in ref_emb else None
    return {"hyp": hyp, "wer": m.wer, "errs": m.substitutions + m.deletions + m.insertions, "words": words,
            "dur": round(dur, 3), "wps": words / max(dur, .1), "max_gap": gap, "peak": float(np.abs(w).max()), "sim": sim}

def flags(v, s, q, med_wps, sim_floor):
    f = []
    if q["words"] < 8 and q["errs"] > 1: f.append(f"short-sentence errors={q['errs']}")
    if q["words"] >= 8 and q["wer"] > .15: f.append(f"wer={q['wer']:.2f}")
    if med_wps and not (.65 * med_wps <= q["wps"] <= 1.5 * med_wps): f.append(f"rate={q['wps']:.2f}wps vs median {med_wps:.2f}")
    if q["max_gap"] > .8: f.append(f"internal silence {q['max_gap']:.2f}s")
    if q["peak"] >= .999: f.append("clipping")
    if q["dur"] < .3: f.append("empty")
    # embeddings of 1-2 s clips are noisy, so short sentences skip the similarity check
    if q["sim"] is not None and q["words"] >= 6 and q["sim"] < sim_floor: f.append(f"speaker sim {q['sim']:.2f}")
    return f

manifest = {"revisions": revs, "voices": {}}
os.makedirs(f"{P}/out", exist_ok=True)
for v in cfg["run_voices"]:
    os.makedirs(f"{P}/out/{v}/wav", exist_ok=True)
    first = []  # first pass: seed 0 takes, measured, to set the per-voice median rate and similarity floor
    items = [(s, cfg["base_seed"] + 1000 * s["n"] + s["idx"]) for s in S]
    for (s, seed), (w, sr, path, gen) in zip(items, synth_many(v, items)):
        first.append((s, seed, w, sr, path, gen, measure(v, s, w, sr)))
        done += 1; progress(voice=v, stage="first-pass")
    med = statistics.median(q["wps"] for *_, q in first)
    sims = [q["sim"] for *_, q in first if q["sim"] is not None]
    if sims:
        ms = statistics.median(sims); mad = statistics.median(abs(x - ms) for x in sims)
        floor = max(.55, ms - 4 * max(mad, .03))  # same voice scored 0.80, a different one 0.14 in calibration
    else:
        floor = -1
    out = []
    T = [[{"seed": seed, "path": path, "gen_s": round(gen, 2), **q, "flags": flags(v, s, q, med, floor)}] for s, seed, w, sr, path, gen, q in first]
    for r in (1, 2, 3):  # retries: every still-flagged sentence together, with its seed + r
        bad = [k for k, tk in enumerate(T) if tk[-1]["flags"]]
        if not bad: break
        items = [(first[k][0], first[k][1] + r) for k in bad]
        for k, (s, sd), (w2, sr2, p2, g2) in zip(bad, items, synth_many(v, items)):
            q2 = measure(v, s, w2, sr2); T[k].append({"seed": sd, "path": p2, "gen_s": round(g2, 2), **q2, "flags": flags(v, s, q2, med, floor)})
    for (s, seed, w, sr, path, gen, q), takes in zip(first, T):
        best = min(takes, key=lambda t: (len(t["flags"]), t["wer"]))
        wb, srb = sf.read(best["path"], dtype="float32")
        sf.write(f"{P}/out/{v}/wav/{s['chapter']}-{s['idx']}.wav", wb, srb)
        out.append({"chapter": s["chapter"], "idx": s["idx"], "text": s["text"], "tts": s["tts"], "chosen_seed": best["seed"],
                    "flags": best["flags"], "takes": takes})
    manifest["voices"][v] = {"median_wps": med, "sim_floor": floor, "sentences": out}
json.dump(manifest, open(f"{P}/out/manifest.json", "w"), indent=1)
progress(finished=True)
print("RUN_DONE", flush=True)

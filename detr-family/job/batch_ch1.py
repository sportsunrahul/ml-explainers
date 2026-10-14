import json, os, time, torch, torchaudio, numpy as np, soundfile as sf, jiwer
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
P = os.path.dirname(os.path.abspath(__file__)) + "/.."
S = [s for s in json.load(open(f"{P}/job/sentences.json")) if s["chapter"] == "ch-set-problem"]
OUT = f"{P}/out/A-batch/wav"; os.makedirs(OUT, exist_ok=True)
m = Qwen3TTSModel.from_pretrained(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base"), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
pr = torch.load(f"{P}/voice/A.prompt.pt", weights_only=False)
torch.manual_seed(1234); torch.cuda.manual_seed_all(1234); t = time.time()
wavs, sr = m.generate_voice_clone(text=[s["tts"] for s in S], language=["English"] * len(S), voice_clone_prompt=pr * len(S))
gen = time.time() - t; dur = sum(len(w) for w in wavs) / sr
print(f"batch {len(S)}: {gen:.1f}s for {dur:.1f}s audio ({gen/dur:.2f}x realtime), vram {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
for s, w in zip(S, wavs): sf.write(f"{OUT}/{s['chapter']}-{s['idx']}.wav", np.asarray(w, dtype=np.float32), sr)
del m; torch.cuda.empty_cache()
from faster_whisper import WhisperModel
from whisper_normalizer.english import EnglishTextNormalizer
from speechbrain.inference.speaker import EncoderClassifier
asr = WhisperModel("large-v3", device="cuda", compute_type="float16"); norm = EnglishTextNormalizer()
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.environ["HF_HOME"] + "/ecapa", run_opts={"device": "cuda"})
r16 = lambda w, sr: torchaudio.functional.resample(torch.from_numpy(w), sr, 16000)
ref, rsr = sf.read(f"{P}/voice/A.ref.wav", dtype="float32"); re_ = enc.encode_batch(r16(ref, rsr)[None].cuda()).squeeze()
gl = ", ".join(json.load(open(f"{P}/lexicon.json")))
rows = []
for tag, d in [("batch", OUT), ("single", f"{P}/out/A/wav")]:
    for s in S:
        w, sr = sf.read(f"{d}/{s['chapter']}-{s['idx']}.wav", dtype="float32")
        segs, _ = asr.transcribe(r16(w, sr).numpy(), language="en", initial_prompt=gl, beam_size=5)
        hyp = " ".join(x.text for x in segs).strip(); dh = lambda x: norm(x.replace("-", ""))
        sim = float(torch.nn.functional.cosine_similarity(re_, enc.encode_batch(r16(w, sr)[None].cuda()).squeeze(), dim=0))
        rows.append({"tag": tag, "idx": s["idx"], "wer": jiwer.wer(dh(s["text"]), dh(hyp)), "sim": sim, "cps": len(s["tts"]) / (len(w) / sr), "hyp": hyp})
for tag in ["batch", "single"]:
    R = [r for r in rows if r["tag"] == tag]
    print(tag, "meanWER %.3f" % np.mean([r["wer"] for r in R]), "sim median %.2f min %.2f" % (np.median([r["sim"] for r in R]), min(r["sim"] for r in R)),
          "chars/s median %.1f" % np.median([r["cps"] for r in R]), "sentences WER>0:", [r["idx"] for r in R if r["wer"] > 0], flush=True)
json.dump(rows, open(f"{P}/out/A-batch/qa.json", "w"), indent=1)
print("BATCH_DONE", flush=True)

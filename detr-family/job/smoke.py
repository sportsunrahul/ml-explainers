import inspect, os, time, json, torch, numpy as np, soundfile as sf
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
OUT = "/root/work/paper-explainer/projects/detr-family/smoke"; os.makedirs(OUT, exist_ok=True)
rep = {}
for n in ["generate_voice_design", "create_voice_clone_prompt", "generate_voice_clone", "generate_custom_voice"]:
    rep[n] = str(inspect.signature(getattr(Qwen3TTSModel, n)))
print(json.dumps(rep, indent=1), flush=True)

def load(mid):
    path = snapshot_download(mid)
    t = time.time()
    m = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    print(mid, os.path.basename(path), f"load {time.time()-t:.1f}s", flush=True)
    return m

REF = "Here is a question worth sitting with. What if a detector simply predicted the set of objects, all at once?"
SENT = "So the model spits out a hundred boxes, but there are only four cars in the picture."
DESC = "A warm, curious male lecturer in his forties, unhurried pace, mid-low pitch, clear diction, a light smile in the voice."

def timed(tag, fn):
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    t = time.time(); wavs, sr = fn(); dt = time.time() - t
    w = np.asarray(wavs[0], dtype=np.float32); sf.write(f"{OUT}/{tag}.wav", w, sr)
    print(tag, f"sr={sr} dur={len(w)/sr:.2f}s gen={dt:.1f}s rtf={dt/(len(w)/sr):.2f} peak={np.abs(w).max():.3f} vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

vd = load("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
timed("ref_design", lambda: vd.generate_voice_design(text=REF, language="English", instruct=DESC))
del vd; torch.cuda.empty_cache()

base = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
prompt = base.create_voice_clone_prompt(ref_audio=f"{OUT}/ref_design.wav", ref_text=REF)
try:
    torch.save(prompt, f"{OUT}/prompt.pt"); print("prompt picklable", flush=True)
except Exception as e:
    print("prompt NOT picklable:", e, flush=True)
timed("clone_a", lambda: base.generate_voice_clone(text=SENT, language="English", voice_clone_prompt=prompt))
timed("clone_b", lambda: base.generate_voice_clone(text=SENT, language="English", voice_clone_prompt=prompt))
del base; torch.cuda.empty_cache()

cv = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
timed("custom_aiden", lambda: cv.generate_custom_voice(text=SENT, language="English", speaker="Aiden", instruct="warm, curious lecturer, unhurried"))
print("SMOKE_DONE", flush=True)

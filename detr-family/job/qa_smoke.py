import os, json, torch, torchaudio, soundfile as sf, numpy as np, jiwer
from faster_whisper import WhisperModel
from whisper_normalizer.english import EnglishTextNormalizer
from speechbrain.inference.speaker import EncoderClassifier
D = "/root/work/paper-explainer/projects/detr-family/smoke"
REF = "Here is a question worth sitting with. What if a detector simply predicted the set of objects, all at once?"
SENT = "So the model spits out a hundred boxes, but there are only four cars in the picture."
norm = EnglishTextNormalizer()
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
def load16(p):
    w, sr = sf.read(p, dtype="float32")
    return torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
for f, ref in [("ref_design", REF), ("clone_a", SENT), ("custom_aiden", SENT)]:
    segs, _ = asr.transcribe(load16(f"{D}/{f}.wav"), language="en", initial_prompt="DETR, COCO", beam_size=5)
    hyp = " ".join(s.text for s in segs).strip()
    print(f, "WER=%.3f" % jiwer.wer(norm(ref), norm(hyp)), "|", hyp, flush=True)
del asr
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.environ["HF_HOME"] + "/ecapa", run_opts={"device": "cuda"})
def emb(p):
    w, sr = sf.read(p, dtype="float32"); w = torch.from_numpy(w)[None]
    w = torchaudio.functional.resample(w, sr, 16000)
    return enc.encode_batch(w.cuda()).squeeze()
r = emb(f"{D}/ref_design.wav")
for f in ["clone_a", "custom_aiden"]:
    print(f, "sim_to_ref=%.3f" % torch.nn.functional.cosine_similarity(r, emb(f"{D}/{f}.wav"), dim=0).item(), flush=True)
print("QA_SMOKE_DONE", flush=True)

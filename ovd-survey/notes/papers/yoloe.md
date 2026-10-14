# YOLOE (branch of YOLO-World, chapter 10; also relevant to chapter 11)

## 1. Identity
- Title: "YOLOE: Real-Time Seeing Anything".
- arXiv 2503.07465, version read: v2. v1 2025-03-10, v2 2025-10-17 (abs.html submission history).
- Authors: Ao Wang (first, equal contribution) ... Guiguang Ding (last, corresponding). Tsinghua University.
- Venue: ICCV 2025 (abs.html comment: "ICCV 2025 Camera-ready Version"; ICCV 2025 template, paper ID 14036).
- Repo: https://github.com/THU-MIG/yoloe (no README fetched).
- Source read: src/camera_ready.tex, paper.pdf for table and equation numbers.

## 2. Core problem
One model should detect and segment from text prompts, visual prompts, or no prompt at all, in real time. Prior work pays for this with neck fusion that grows with vocabulary size (YOLO-World, Grounding DINO), transformer-heavy visual-prompt encoders (T-Rex2), or a language model that generates names (GenerateU, DINO-X) (Sec. 1).

## 4. The one key idea
Keep a plain YOLO (v8 or 11) and contrast each anchor point's object embedding with a prompt embedding (Eq. 1: `Label = O . P^T`). Each prompt type only changes how `P` is made: text through a small FFN that is folded into the last 1x1 conv after training (RepRTA), a box or mask through a light two-branch encoder (SAVPE), and no prompt through one learned "find everything" embedding plus retrieval from a 4585-name vocabulary (LRPC). Where text enters: at the classifier only. YOLOE removes YOLO-World's cross-modal neck fusion (Table 5).

## 5. Mechanism (at most 5 steps)
1. Architecture (Sec. 3.1): backbone, PAN, regression head, segmentation head (YOLACT-style prototypes + coefficients), object embedding head (the YOLO classification head with the last 1x1 conv outputting D channels instead of class count). Text prompts use the MobileCLIP-B(LT) text encoder (Sec. 4.1).
2. RepRTA, text (Sec. 3.2, Eq. 2-3): cache CLIP text embeddings `P` before training (the text encoder is not run in training); one SwiGLU FFN block `f_theta` refines them. After training: `K' = R(f_theta(P)) (conv) K^T`, so `Label = I (conv) K'`, an architecture "identical to the original YOLO", zero deployment overhead.
3. SAVPE, visual (Sec. 3.3, Eq. 4): the prompt is a 0/1 mask. An activation branch with A = 16 channels fuses mask and image features into weights, softmax-normalized inside the prompt region. A semantic branch makes D-channel features from P3-P5. Each of the A weight maps pools its D/A-channel group; the groups are concatenated into `P`. Trained 2 epochs with all other parts frozen.
4. LRPC, prompt-free (Sec. 3.4, Eq. 5): one specialized embedding `P_s` is trained (1 epoch) to fire on any object; only anchors with `o . P_s^T > delta` (delta = 0.001) are matched against a built-in vocabulary of 4585 names (RAM tag list). No language model.
5. Training (Sec. 3.5, 4.1): online vocabulary per mosaic as in YOLO-World; task-aligned assignment; binary cross entropy for classification, IoU + DFL for boxes. Data: Objects365v1 + GoldG, with SAM-2.1 pseudo masks. Text stage 30 epochs (YOLO-World: 100), on 8 RTX4090 GPUs.

## 8. Key table
Table 5, "Roadmap to YOLOE in terms of text prompts": YOLO-Worldv2-L -> YOLOE-v8-L. Standard AP (not Fixed AP) on LVIS minival, zero-shot; FPS on T4 with TensorRT.

| Step (cumulative) | AP | FPS T4 |
|---|---|---|
| YOLO-Worldv2-L, 30 epochs + global negative dictionary (fusion still in) | 31.9 | 80.0 |
| - cross-modal fusion | 30.0 | 102.5 |
| + MobileCLIP text encoder | 31.5 | 102.5 |
| + RepRTA | 33.5 | 102.5 |

Reading: removing the neck fusion cost 1.9 AP (text) and bought 1.28x speed on T4 (text). A better frozen text encoder plus a foldable FFN beat the fused model at no inference cost.

## 10. Outline feedback
1. The chapter 10 one-liner could be: "YOLOE (2025) drops the neck fusion again: text enters only at the classifier, through a small FFN folded into the last 1x1 conv; the same head takes box prompts (SAVPE) or no prompt (LRPC)." This is a counterpoint to YOLO-World's "fuse into the neck".
2. Color YOLOE "classifier only" in the family tree, not "neck".
3. Its baseline is YOLO-Worldv2 (repo model), not the v3 paper's YOLO-World; and Table 5 uses standard AP while Table 1 uses Fixed AP. Do not put these next to YOLO-World Table 2 numbers without saying so.
4. Chapter 11: Sec. 4.2 says SAVPE visual prompts reach APr and APc comparable to text prompts, the same observation as T-Rex2. That is a second source for "boxes help where names fail" (numbers in Table 1 if needed).

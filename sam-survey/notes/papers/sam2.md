# SAM 2: Segment Anything in Images and Videos

## 1. Identity
- Title: "SAM 2: Segment Anything in Images and Videos".
- arXiv 2408.00714, version read: v2 (LaTeX file `sam2.1_arxiv.tex`; paper.pdf is 42 pages and matches the tex).
- Dates (abs.html): v1 2024-08-01, v2 2024-10-28.
- Authors: Nikhila Ravi ... Christoph Feichtenhofer (18 authors, Meta FAIR). Project leads: Ravi, Dollar, Feichtenhofer.
- Venue: not stated in the text. Hint only: the bib style is `iclr2025_conference` and a table comment says "for ICLR submission".
- Repo: https://github.com/facebookresearch/sam2 (Apache 2.0). SA-V is CC BY 4.0.
- Version note (Sec. 1 footnote): "All the results presented in this paper are based on a new version of SAM 2 (improved over our initial release; denoted as 'SAM 2.1' ...), which we will refer to as SAM 2". So every v2 number is SAM 2.1.
- The only training change the text describes that matches 2.1 is the 16-frame fine-tuning stage (App. D.2.2). The text does not label it "the 2.1 change". The tab/*.tex files keep the SAM 2.0 rows as comments ("reverting to SAM 2.0 results for ICLR submission" ... "Below are the new SAM 2.1 results"). The active rows are 2.1.

## 2. Core problem
SAM segments objects in single images only. Video adds motion, deformation, occlusion, blur and many frames. Existing video segmentation models and datasets cover a few object classes and whole objects only, not "anything", and they have no cheap way to correct a track interactively.

## 3. What prior methods did and why that hurt (per the paper, Sec. 2)
- Semi-supervised VOS (STM/XMem/Cutie family): the input is a mask on the first frame only. A high-quality first-frame mask is "practically challenging and time-consuming" to get.
- SAM + tracker (Track Anything, SAM-PT, DEVA style): "the tracker may not work for all objects, SAM may not perform well on video frames, and there is no mechanism to interactively refine a model's mistakes, other than re-annotating using SAM in each frame and restarting the tracking from there" (Sec. 2). Fig. 2: a decoupled SAM + tracker needs several clicks to recover a lost object; SAM 2 needs one, because of memory.
- VOS datasets (DAVIS, YouTube-VOS, MOSE, ...) cover whole objects of specific classes (people, vehicles, animals) and are small.

## 4. The one key idea
Treat an image as a one-frame video. Run SAM's promptable decoder on each frame, but first condition that frame's features on a streaming memory of past predictions and prompted frames (memory attention). A prompt on any frame then corrects the whole masklet. When the memory is empty, the model behaves like SAM.

## 5. Mechanism, step by step
1. PVS task (Sec. 3, App. B): prompts (positive/negative clicks, boxes, masks) can come on any frame. The model must answer at once with a valid mask on that frame, then propagate to a masklet over the whole video. More prompts on any frame refine it. SA (images) and semi-supervised VOS (one mask on frame 1) are special cases (Fig. 7).
2. Image encoder (Sec. 4, App. D.1): MAE-pretrained Hiera, run once per frame (streaming). An FPN fuses the stride-16 (stage 3) and stride-32 (stage 4) features into the frame embedding (256x64x64 at 1024 input, Fig. 8 label). The stride-4 (stage 1) and stride-8 (stage 2) features do NOT go into memory attention. They go to the decoder's upsampling layers as skip connections. No relative position bias (RPB). The encoder uses windowed absolute position embeddings, with the global embedding interpolated across windows. Global attention is in a few blocks only (Table 12: 5-7-9 T, 7-10-13 S, 12-16-20 B+, 23-33-43 L).
3. Memory attention (Sec. 4, App. D.1): L = 4 transformer blocks. Each block: self-attention -> cross-attention to the memory bank (spatial memories + object pointers) -> MLP. Vanilla attention (so FlashAttention-2 works). Sinusoidal absolute PE plus 2d-RoPE in self- and cross-attention. Object pointer tokens are excluded from RoPE. The ablation grid (#sa, #ca) default is (4, 4) (Table 9e). Our reading: "4 layers" = 4 self-attention + 4 cross-attention.
4. Prompt encoder: identical to SAM's. Sparse prompts = positional encoding + learned type embedding. Mask prompts = convs, summed with the frame embedding.
5. Mask decoder (Sec. 4, Fig. 8): SAM's two-way transformer ("x2" in Fig. 8), multiple masks for ambiguous prompts. Changes vs SAM:
   - Occlusion head: an extra output token (next to the mask and IoU tokens) + MLP gives a score that the object is visible on this frame. It handles "no valid object on this frame".
   - Skip connections: stride-4 and stride-8 encoder features are added during the 2x transposed-conv upsampling.
   - Object pointer: the mask token of the chosen output mask becomes the frame's object pointer (256-dim).
   - In video, if no later prompt resolves the ambiguity, only the mask with the highest predicted IoU propagates.
6. Memory encoder (Sec. 4, D.1): downsample the output mask with a conv module, sum it element-wise with the unconditioned frame embedding from the image encoder, then fuse with light conv layers. No separate image encoder. Memory features are projected to 64 channels. A learned occlusion embedding is added to the memories of frames the occlusion head predicts as occluded.
7. Memory bank (Sec. 4): a FIFO of up to N recent (unprompted) frame memories and a FIFO of up to M prompted frame memories, both spatial maps. Default N = 6 ("a default value of 6 past frames", App. A.2.1). M is NOT stated in the paper or README. In VOS (one mask prompt) the bank keeps frame 1's memory + up to N recent. Temporal position encoding goes on the N recent memories only, not on prompted ones. Object pointers: the 256-dim pointer is split into 4 tokens of 64 dims for cross-attention. The number of stored pointers is not stated.
8. Multi-object (D.1, App. C): image-encoder features are shared across objects. The memory bank, memory attention and decoder run separately per object. There is no inter-object communication (listed as a limitation).

### Shapes for the ArchFlow stepper (1024x1024 input)
| tensor | shape | source |
|---|---|---|
| input frame | 3x1024x1024 | Table 12 (resolution 1024) |
| Hiera stage 1 (stride 4) | C1 x256x256 | stride: D.1; spatial = 1024/4, our reading |
| Hiera stage 2 (stride 8) | C2 x128x128 | as above |
| Hiera stage 3 (stride 16) | C3 x64x64 | as above |
| Hiera stage 4 (stride 32) | C4 x32x32 | as above |
| FPN frame embedding (stages 3+4) | 256x64x64 | Fig. 8 label "image embedding (256x64x64)" |
| prompt + output tokens | N_tokens x256 | Fig. 8 label |
| memory per frame | 64 x64x64 | 64 channels: D.1; 64x64 spatial is our reading |
| object pointer per frame | 4 tokens x 64 | D.1 (256 split into 4x64) |
| memory attention input/output | 256x64x64 | our reading (consistent with Fig. 8) |
| decoder upsampling | 2x conv-trans twice, adding stride-8 then stride-4 features | Fig. 8, D.1; ordering is our reading |

Hiera channel widths (from the Hiera paper, arXiv 2306.00989v1, Table "Configuration for Hiera variants", not the SAM 2 paper):
T [96-192-384-768] blocks [1-2-7-2]; S [96-192-384-768] [1-2-11-2]; B+ [112-224-448-896] [2-3-16-3] heads [2-4-8-16]; L [144-288-576-1152] [2-6-36-4] heads [2-4-8-16].
Our reading: the global-attention indices in Table 12 are 0-based and all fall in stage 3 (B+ has 24 blocks, stage 3 = blocks 5-20). Hiera's 70M (B+) and 214M (L) are encoder-only params at 224 input. README's 80.8M / 224.4M are whole SAM 2.1 checkpoints. Do not equate them.

### Training
- Pre-training on SA-1B images (D.2.1, Table 12a): ~90k steps, batch 256, AdamW (0.9, 0.999), lr 4e-4, reciprocal-sqrt schedule (timescale 1000), 1k warmup, 5k cooldown, wd 0.1, grad clip l2 0.1, layer decay 0.8 (T,S) / 0.9 (B+) / 0.925 (L), drop path 0.1/0.2/0.3, bf16, hflip, 64 masks per image (masks >90% of image filtered).
- Deltas vs SAM (D.2.1): IoU head gets an l1 loss and a sigmoid on its logits (SAM used MSE). IoU is supervised for ALL masks, the mask loss only for the lowest-loss mask. No "no-new-prompt" iterations (SAM inserted 2). 7 correction clicks (SAM: 8).
- Full training (D.2.2, Table 12b): ~150k steps, batch 256, lr img. enc. 6e-5 / other 3.0e-4, cosine, 7.5k warmup. Joint image + video: each iteration samples a whole batch from images OR videos, with probability proportional to source size. Mix: ~15.2% SA-1B, ~70% SA-V, ~14.8% Internal. With the OSS sets: ~1.3% DAVIS, ~9.4% MOSE, ~9.2% YouTubeVOS, ~15.5% SA-1B, ~49.5% SA-V, ~15.1% Internal. Manual masklets only (no auto). "Our released model is trained on SA-V manual + Internal and SA-1B."
- Simulated interaction (Sec. 4, D.2.2): 8-frame sequences. Up to 2 frames (including the first) get prompts / corrective clicks. Initial prompt: GT mask p=0.5, positive click p=0.25, box p=0.25. Max 3 masklets per sequence. 50% temporal reversal. 10% of corrective clicks are sampled from the GT mask, ignoring the prediction.
- Video augmentation: hflip, affine (deg 25, shear 20), color jitter, grayscale 0.05, per-frame jitter, mosaic 2x2 with p=0.1 (same video tiled; the target is in one quadrant, so the model must use motion).
- Losses (D.2.2): focal : dice : IoU (MAE/l1) : occlusion cross-entropy = 20:1:1:1. For multi-mask outputs, only the lowest-loss mask is supervised. On frames with no GT mask, no mask output is supervised, but the occlusion head always is.
- 16-frame fine-tuning (D.2.2): 16-frame sequences from the top 50% most-edited masklets (SA-V, Internal), full OSS sets kept. 50k iterations (1/3 schedule), half lr, image encoder frozen to fit 80 GB A100 memory.
- Compute (Table 18 model card): released SAM 2 trained on 256 A100 GPUs for 108 hours (12165.12 kWh, 3.89 t CO2e).

## 6. Equations
The paper has no numbered equations and no loss equation (the loss is given only as the 20:1:1:1 ratio). Macros `\imageEmbed`, `\memBank`, etc. are defined but unused. The only formula is the annotation-time model of App. F.1.2 (verbatim):

Offline: `$\left(T_\mathrm{exam} \cdot \left(L/300\right) + T_\mathrm{loc} + T_\mathrm{click} \cdot N_\mathrm{click}\right) \cdot N_\mathrm{frame}$`
Online: `$T_\mathrm{exam} \cdot \left(L/300\right) + \left(T_\mathrm{loc} + T_\mathrm{click} \cdot N_\mathrm{click}\right) \cdot N_\mathrm{frame}$`

Meaning: the estimated human time to annotate one object. Offline re-examines the whole video every round; online examines it once.
- T_exam = 30 s to examine a 300-frame video (10 FPS viewing). L = total frames in the video. T_loc = 1 s to locate the object on a frame. T_click = 1.5 s per click. N_click = 3 clicks per frame. N_frame = 1..8 interacted frames.

Macros expanded for numbers: `\savinumvideos`=50.9K, `\savinummasklets`=190.9K, `\savinummaskletsAutoOnly`=451.7K, `\savinummaskletsWAuto`=642.6K, `\savinummasks`=10.0M, `\savinummasksWAuto`=35.5M, `\savifpsVideoFrames`=4.2M, `\saviVideoDuration`=196.0 hr, `\saviDisappRate`=42.5%, `\saviDisappRateWAuto`=27.7%, `\saviMoreMasksWAutoRate`=53x, `\saviMoreMasksRate`=15x, phase masklets 16K / 63.5K / 197.0K, val 155 videos / 293 masklets, test 150 / 278, internal 62.9K videos / 69.6K masklets / 5.4M masks / 6.0M frames / 281.8 hr / 36.4%.

## 7. Config / setup table
| item | value | loc |
|---|---|---|
| input resolution | 1024 (ablations at 512) | Table 12; App. A |
| encoder | Hiera T / S / B+ / L, MAE init | Sec. 4, D.1 |
| default model | Hiera-B+, 1024 | Sec. 6 |
| FPN inputs | stride 16 + 32 (stages 3, 4) | D.1 |
| decoder skips | stride 4 + 8 (stages 1, 2) | D.1, Fig. 8 |
| memory attention | L = 4; self -> cross -> MLP; 2d-RoPE | Sec. 4, D.1 |
| memory channels | 64 (ablation: 256 costs 0.92x speed) | D.1, Table 9d |
| object pointer | 256-dim -> 4 x 64 tokens | D.1 |
| N recent memories | 6 | A.2.1, Table 9c |
| M prompted memories | not stated | -- |
| train sequence | 8 frames (16 in fine-tune) | Sec. 4, D.2.2 |
| prompted frames per sequence | up to 2 | Sec. 4 |
| masklets per sequence | up to 3 | D.2.2, Table 12b |
| loss weights | focal 20, dice 1, IoU l1 1, occlusion CE 1 | Table 12b |
| params / speed (2.1) | T 38.9M 91.2 FPS; S 46M 84.8; B+ 80.8M 64.1; L 224.4M 39.5 | README (code-only; A100, torch 2.5.1, full compile) |
| speed (paper, video) | B+ 43.8 FPS, L 30.2 FPS | Sec. 7 (A100, batch 1) |
| speed (paper, image SA task) | B+ 130.1 FPS vs SAM ViT-H 21.7 | Table 5/15 (A100, batch 10, encoder compiled, torch 2.3.1, D.3) |

## 8. Results worth remembering
Data engine (Table 1, controlled experiment on 169 videos / 452 masklets, App. E.2.2):
| phase | model in loop | time/frame | edited frames | clicks per clicked frame | P1 alignment All/S/M/L |
|---|---|---|---|---|---|
| 1 | SAM only | 37.8 s | 100.00 % | 4.80 | - |
| 2 | SAM + SAM 2 Mask | 7.4 s | 23.25 % | 3.61 | 86.4 / 71.3 / 80.4 / 97.9 % |
| 3 | SAM 2 | 4.5 s | 19.04 % | 2.68 | 89.1 / 72.8 / 81.8 / 100.0 % |
- The alignment score is the percent of masks with IoU > 0.75 vs Phase 1. Speed-ups are ~5.1x (P2) and ~8.4x (P3) over P1. Phase 2 retrained SAM 2 Mask twice, Phase 3 retrained SAM 2 five times. Masklets: P1 16K (1.4K videos), P2 63.5K, P3 197.0K (Sec. 5.1).
- Flag: 16K + 63.5K + 197.0K > 190.9K SA-V manual masklets. The paper does not explain this. Our reading: Internal (69.6K masklets, Phases 2-3) may account for part of it. Do not sum.
- Table 2 (fixed iterations, 3-click J&F): VOS+SA-1B 50.0 / 62.5; +P1 53.0 / 66.9; +P2 58.8 / 70.9; +P3 62.5 / 71.2; +Auto 63.2 / 71.5 (SA-V val / 9 zero-shot).
- Quality verification (Sec. 5.1): separate annotators mark masklets "satisfactory" or "unsatisfactory". Unsatisfactory ones go back for refinement. Masklets of not-well-defined objects are rejected.
- Auto masklets (Sec. 5.1, E.1): a 32x32 grid on frame 1, plus 16x16 on 4 crops and 4x4 on 16 crops. Components and holes < 200 px are removed or filled. Satisfactory auto masklets join SA-V. Unsatisfactory ones go to Phase 3 annotators.

SA-V (Sec. 5.2, Table 3, E.1): 50.9K videos, 642.6K masklets (190.9K manual + 451.7K auto), 35.5M masks (10.0M manual), 4.2M frames, 196 hr. 54% indoor / 46% outdoor. Average duration 14 s (Sec. 5.2) vs 13.8 s (E.1); both are stated. Range 4 s - 2.3 min. Resolution 240p - 4K, average 1,401 x 1,037. 47 countries. >88% of masks have normalized area < 0.1. Annotated at 6 FPS, videos resampled to 24 fps. Disappearance rate 42.5% (manual), 27.7% (with auto). 53x more masks than any VOS dataset (15x without auto). Val 155 videos / 293 masklets, test 150 / 278, both annotated with the Phase 1 protocol.
Table 3 comparison (videos, duration, masklets, masks, frames, disapp.): DAVIS 2017 0.2K, 0.1 hr, 0.4K, 27.1K, 10.7K, 16.1%; YouTube-VOS 4.5K, 5.6 hr, 8.6K, 197.3K, 123.3K, 13.0%; UVO-dense 1.0K, 0.9 hr, 10.2K, 667.1K, 68.3K, 9.2%; VOST 0.7K, 4.2 hr, 1.5K, 175.0K, 75.5K, 41.7%; BURST 2.9K, 28.9 hr, 16.1K, 600.2K, 195.7K, 37.7%; MOSE 2.1K, 7.4 hr, 5.2K, 431.7K, 638.8K, 41.5%; Internal 62.9K, 281.8 hr, 69.6K, 5.4M, 6.0M, 36.4%.

Zero-shot:
- PVS (Fig. 5, 9 dense datasets, 3 clicks/frame, average over 1-8 interacted frames): SAM 2 80.3 offline / 79.8 online vs SAM+Cutie 74.7 / 74.0 vs SAM+XMem++ 71.7 / 72.8 (Fig. 12b, 13b "average" column). Text: ">3x fewer interactions" (Sec. 6.1); the abstract says "3x". The curve values in Fig. 5 cannot be extracted from the source.
- Semi-supervised VOS, 17 datasets (Table 4), 1-click / 3-click / 5-click / box / GT mask: SAM 2 64.7 / 75.3 / 77.6 / 74.4 / 79.3; SAM+Cutie 56.7 / 70.1 / 72.2 / 69.4 / 74.1; SAM+XMem++ 56.9 / 68.4 / 70.6 / 67.6 / 72.7.
- Image SA task, 37 datasets (Table 5 / 15), 1-click (5-click) mIoU on SA-23 All: SAM ViT-H 58.1 (81.3) at 21.7 FPS; SAM 2 B+ SA-1B-only 58.9 (81.7) at 130.1 FPS; SAM 2 B+ our mix 61.9 (83.5). 14 new video datasets: SAM 59.1 -> SAM 2 mix 69.6. Hiera-L mix 63.6 at 61.4 FPS. SAM ViT-B 55.9 at 76.7 FPS. HQ-SAM ViT-H 59.1 (79.8) at 21.4 FPS. SAM 2 B+ beats SAM on 29 of 37 datasets, by up to 53.9 mIoU (F.3.2).
- "6x faster": 130.1 vs 21.7 FPS (Table 5).
- Flag: the prose says "61.4%" for the mix on SA-23 (Sec. 6.3, F.3.2), but Tables 5/15 say 61.9. Prose also says Hiera-L is "3.4x faster than SAM (ViT-H) and 1.5x faster than SAM (ViT-B)" (F.3.2). Table 15 gives L 61.4 vs 21.7 / 76.7 FPS, which does not give those ratios. Our reading: this prose is left over from v1.
- SOTA VOS (Table 6, GT mask on frame 1), J&F except YTVOS (G):
  | method | MOSE val | DAVIS17 val | LVOS val | SA-V val | SA-V test | YTVOS19 val |
  |---|---|---|---|---|---|---|
  | Cutie-base+ | 71.7 | 88.1 | - | 61.3 | 62.8 | 87.5 |
  | Cutie-base | 69.9 | 87.9 | 66.0 | 60.7 | 62.7 | 87.0 |
  | XMem | 59.6 | 86.0 | - | 60.1 | 62.3 | 85.6 |
  | JointFormer | - | 90.1 | - | - | - | 87.4 |
  | SAM 2 B+ | 76.6 | 90.2 | 78.0 | 76.8 | 77.0 | 88.6 |
  | SAM 2 L | 77.9 | 90.7 | 78.0 | 77.9 | 78.4 | 89.3 |
  Flag: LVOS val B+ and L rows are identical (78.0 / 73.2 / 82.7) in both the tex and the PDF (Table 17b). This may be a copy error in the source.
- Table 6 SAM 2 rows use the full mix (with OSS VOS sets). The double-dagger rows of Table 17 (SA-1B + SA-V + Internal = the released mix) match the README 2.1 numbers: SA-V test T 76.5, S 76.6, B+ 78.2, L 79.5; LVOSv2 T 77.3, S 78.3, B+ 78.2, L 80.6. One mismatch: MOSE val B+ is 73.8 in Table 17d but 73.7 in the README.
- SAM 2.0 -> 2.1 (README, code-only): SA-V test L 76.0 -> 79.5, B+ 74.7 -> 78.2; LVOSv2 L 79.8 -> 80.6.
- DAVIS interactive (Table 14): SAM 2 (clicks) AUC 0.86, J&F@60s 0.90 vs CiVOS 0.83 / 0.84.

Ablations (512 res, SA-V manual + 10% SA-1B, 3-click J&F):
- #memories N (Table 9c): 4 / 6 / 8 -> MOSE dev 73.5 / 73.0 / 73.2; SA-V val 68.6 / 68.3 / 69.0; 9 zero-shot 70.5 / 70.7 / 70.7; speed 1.01x / 1.00x / 0.93x.
- Memory channels (Table 9d): 64 vs 256 -> SA-V val 68.3 vs 66.4, speed 1.00x vs 0.92x. 64 channels make storage 4x smaller.
- Memory attention (#sa, #ca) (Table 9e): (2,2) 67.3 SA-V val, 1.13x; (3,2) 64.1, 1.08x; (4,4) 68.3, 1.00x.
- #frames (Table 9b): 4 / 8 / 10 -> SA-V val 60.0 / 68.3 / 68.1.
- Resolution (Table 9a): 512 / 768 / 1024 -> speed 1.00x / 0.43x / 0.22x; SA-23 59.7 / 61.0 / 61.5.
- Encoder (Table 9f): S / B+ / L -> SA-V val 65.5 / 68.3 / 66.3; SA-23 57.8 / 59.7 / 61.1; speed 1.33x / 1.00x / 0.60x.
- Positional encoding (Table 10): 2d-RoPE without RPB (default) 68.3 SA-V val, LVOSv2 71.6; no RoPE 67.1 / 70.3 at 1.04x speed.
- Object pointers / GRU (Table 11), SA-V val / LVOSv2 val / 9 zero-shot: none 64.5 / 67.0 / 70.9; GRU 65.3 / 68.9 / 70.5 (0.97x); object pointers (default) 68.3 / 71.6 / 70.7. Recurrence (GRU) does not help, except slightly on LVOSv2.
- Data mix (Table 7, 200k iter, batch 128): VOS only gives 59.7 on 9 zero-shot. Adding data-engine data gives +12.1 (row 11, 71.8). The best overall is all data (row 12: SA-V val 63.1, MOSE dev 79.0, SA-23 58.9).
- Data quality (Table 8): 50k random 63.7 vs 50k most-edited 66.2 vs all 190k 69.9 (SA-V val).
- Fairness (Table 13): 3-click J&F male 95.1, female 94.1.

## 9. Limitations (App. C, quoted)
- "The model may fail to segment objects across shot changes and can lose track of or confuse objects in crowded scenes, after long occlusions or in extended videos."
- "SAM 2 also struggles with accurately tracking objects with very thin or fine details especially when they are fast-moving."
- "Another challenging scenario occurs when there are nearby objects with similar appearance (e.g., multiple identical juggling balls). Incorporating more explicit motion modeling into SAM 2 could mitigate errors in such cases."
- "SAM 2 processes each object separately, utilizing only shared per-frame embeddings without inter-object communication."
- "Our data engine relies on human annotators to verify masklet quality and select frames that require correction."
- Later fixes: long videos and occlusion map to SAM2Long; similar objects and motion modeling map to SAMURAI; inter-object sharing maps to SAM 3's "Object Multiplex" appendix per the outline. These are not verified from those sources here.

## 10. Outline feedback (ch-sam2)
- Memory bank: N = 6 is sourced. M (the prompted-frame FIFO size) is not in the paper or README, so the scene should not show a number for M. The number of stored object pointers is also not stated.
- The ArchFlow should show three things the outline omits. First, the stride-4/8 skip path that bypasses memory attention into the decoder. Second, the object pointer (mask token -> 4 x 64 tokens) feeding the memory bank. Third, the occlusion token whose prediction adds a learned occlusion embedding to that frame's memory.
- Temporal position encoding goes on recent memories only, not prompted ones. This is a good visual detail.
- Data-engine claim card: use Table 1 (37.8 / 7.4 / 4.5 s per frame; edited frames 100 / 23.25 / 19.04 %; clicks 4.80 / 3.61 / 2.68). Do not sum the phase masklet counts.
- "Image = video with empty memory" has a concrete result: SAM 2 B+ (SA-1B only) beats SAM ViT-H on SA-23 (58.9 vs 58.1) at 130.1 vs 21.7 FPS. This links back to ch-sam-arch. Say "6x" only for the image task. Video speed is 43.8 FPS (B+).
- The interlude ch-training (SAM) could note SAM 2's changes: 7 correction clicks (vs 8), no empty rounds (vs 2), l1 IoU loss on all masks. In video: 8 frames, up to 2 prompted, GT-mask/click/box = 0.5/0.25/0.25.
- Strongest memory-design ablation for a claim card: object pointers (Table 11). GRU recurrence gives nothing.
- Avoid these prose numbers: "61.4%" (use the table value 61.9) and "3.4x / 1.5x faster" (inconsistent with Table 15).

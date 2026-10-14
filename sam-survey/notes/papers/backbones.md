# Backbones SAM inherits: ViT -> MAE -> ViTDet; Hiera (SAM 2)

Scope: short backbone facts for ch-sam-arch (ViT-H, MAE init, windowed + global attention) and ch-sam2 (Hiera).

---------------------------------------------------------------------------------------------------
## ViT -- An Image is Worth 16x16 Words

### 1. Identity
Dosovitskiy (first) ... Houlsby (last). arXiv 2010.11929v2 (v1 2020-10-22). The reader already has a ViT explainer; only the config is extracted.

### 7. Config (Table 1, "Details of Vision Transformer model variants")
| model | Layers | Hidden size D | MLP size | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |
- Notation (Sec. 4.1 "Model Variants"): "ViT-L/16 means the 'Large' variant with 16x16 input patch size." Table 1 does not depend on patch size.
- "Huge" is new in ViT; Base and Large come from BERT (Sec. 4.1).
- ViT's largest model is ViT-H/14 (Sec. 4.2, Table 2). Fine-tuned at 518 for H/14 (Sec. 4.1 "Training & Fine-tuning").
- Head dim for ViT-H = 1280 / 16 = 80 (our arithmetic).

---------------------------------------------------------------------------------------------------
## MAE -- Masked Autoencoders Are Scalable Vision Learners

### 1. Identity
He (Kaiming, first) ... Girshick (last). arXiv 2111.06377v3 (v1 2021-11-11). abs.html: "Tech report. arXiv v2: add more transfer learning results; v3: add robustness evaluation".

### 2-4. Problem and key idea
Images are spatially redundant, so masking a few patches is too easy. Mask a very high ratio (75%) of random patches; run the encoder only on the visible 25%; a light decoder with mask tokens reconstructs pixels (Sec. 1, Sec. 3).

### 5. Mechanism (Sec. 3)
1. Patchify; random sampling without replacement, uniform; masking ratio 75%.
2. Encoder = ViT on visible patches only; "no mask tokens are used."
3. Decoder input = encoded visible tokens + shared learned mask token at each masked position, with positional embeddings. Default decoder: 8 blocks, width 512 (Table 1 caption; Sec. 4.1).
4. Loss: MSE in pixel space, "only on masked patches" (Sec. 3, "Reconstruction target"). Variant: per-patch normalized pixels (better). No equation printed.
5. After pretraining the decoder is discarded; the encoder sees full images (Fig. 1 caption).
6. Skipping mask tokens in the encoder: 3.3x fewer training FLOPs, 2.8x wall-clock speedup; 3.5-4.1x for ViT-H or a 1-block decoder (Sec. 4.1 "Mask token", Table 2).

### 7. Config
| item | value | loc |
|---|---|---|
| masking ratio | 75% | abstract, Fig. 5, Sec. 4.1 |
| decoder | 8 blocks x 512-d (9% FLOPs per token vs ViT-L) | Sec. 4.1 "Decoder design" |
| ViT-H used | H/14 ("The ViT models are B/16, L/16, H/14") | Table 3 caption |
| ViT-H params | 632 M | App., Table 12 (linear probing) |
| ViT-H IN-1K fine-tune | 86.9 (224), 87.8 (448) | Table 3 |
| pretraining length (Table 3) | 1600 epochs, normalized pixels | Table 3 caption |

### Param-count note
- MAE Table 12: ViT-H "632 M". ViT Table 1: ViT-Huge "632M".
- SAM Fig. 13 (right) x-axis label: ViT-H "636M" (already claimed in sam-model.json / sam-data.json).
- Not reconciled. Our reading: SAM likely counts extra parts (e.g. its neck, patch 16 vs 14), but SAM does not say what is counted.

---------------------------------------------------------------------------------------------------
## ViTDet -- Exploring Plain Vision Transformer Backbones for Object Detection

### 1. Identity
Li (Yanghao, first), Mao, Girshick, He (Kaiming, last). arXiv 2203.16527v2 (v1 2022-03-30). abs.html: "Tech report. arXiv v2: add RetinaNet results". Code: detectron2/projects/ViTDet.

### 2-4. Problem and key idea
Detection wants high-resolution input and multi-scale features. Plain ViT has one scale and global attention is too costly at 1024. Keep the pretrained plain ViT; during fine-tuning only, use window attention plus a few cross-window blocks, and a simple feature pyramid from the last map (abstract, Sec. 1).

### 5. Mechanism (Sec. 3)
1. Input 1024x1024 (Sec. 4), patch 16 -> 64x64 tokens, stride 16.
2. Window attention, non-overlapping, no shift. Window size = pretraining feature map size, "14x14" by default (Sec. 3 footnote).
3. Propagation: split the backbone evenly into 4 subsets; the last block of each subset uses global self-attention (or a conv block). Default 4 blocks, evenly placed (Sec. 3, Sec. 4.1). Example given only for ViT-L: "6 in each subset for the 24-block ViT-L".
   - For 32-block ViT-H: 8 per subset, global at blocks 8, 16, 24, 32 (1-based). Our reading; ViTDet does not print it.
4. Simple feature pyramid: from the last 1/16 map, parallel conv/deconv with strides {2, 1, 1/2, 1/4} -> scales {1/32, 1/16, 1/8, 1/4} (Sec. 3). 1/32 by 2x2 max pool; 1/8 and 1/4 by one or two 2x2 stride-2 deconvs; then 1x1 conv + LN to 256 ch and 3x3 conv + LN per level (App. A.2 "Architectures").
5. Patch-size hop: "As ViT-H in [ViT] by default has a patch size of 14, after pre-training we interpolate the patch embedding filters from 14x14x3 to 16x16x3." (App. A.2). ViTDet uses patch 16 for B, L, H.
6. Relative position biases added only during fine-tuning in system-level comparisons (Sec. 4.2); ablations are without them.

### 8. Results
- Ablation, ViT-L Mask R-CNN COCO (Table 2): window only 52.9 AP box; 4 global blocks 54.6; 24 global 55.1 (needs memory checkpointing).
- Feature pyramid, Mask R-CNN COCO AP box (Table 1): no pyramid ViT-B 47.8 / ViT-L 51.2; simple feature pyramid ViT-B 51.2 / ViT-L 54.6 (same as FPN variants).
- ViT-H Cascade, system level: 61.3 AP box COCO (abstract, Table 6).

---------------------------------------------------------------------------------------------------
## Hiera -- A Hierarchical Vision Transformer without the Bells-and-Whistles

### 1. Identity
Ryali (first) ... Li (Yanghao), Feichtenhofer (last). arXiv 2306.00989v1 (2023-06-01). ICML 2023 Oral (abs.html). Repo: github.com/facebookresearch/hiera.

### 2-4. Problem and key idea
Hierarchical ViTs (Swin, MViTv2) add vision-specific modules that make them slower than plain ViT. With MAE pretraining the model can learn spatial bias instead, so strip the modules from MViTv2 (abstract, Sec. 1).

### 5. Mechanism
1. 4 stages; tokens start at 4x4 pixels; at each stage transition Q and the skip are doubled in channels by a linear layer and pooled by 2x2 maxpool (Fig. 2 caption). Stage resolutions at 224: [56^2, 28^2, 14^2, 7^2] (Table 2 caption). Strides 4/8/16/32 = our reading from these.
2. Attention: "Mask Unit attention" (local, within mask units) in stages 1-2; global attention in stages 3-4 (Fig. 2 caption, Fig. 5 caption).
3. MAE pretraining with 32x32-pixel mask units (8x8 tokens at stage 1). Mask ratio 0.6 for images, 0.9 for video (Table 3, mask ratio sub-table). Paper: 0.6 < MAE's 0.75 "due to ... a 32x32 mask unit instead of 16x16".
4. Removed vs MViTv2 (Table 1, rows a-f): relative position embeddings -> absolute; convs -> maxpools; stride-1 maxpools deleted; kernel size = stride; Q attention residuals deleted; KV pooling -> mask unit attention. Summary (Sec. 1): "no convolutions, no shifted or cross-shaped windows, no decomposed relative position embeddings."
5. Hiera-L is 2.4x faster on images and 5.1x on video than the starting MViTv2 (end of Sec. 3.2).

### 7. Config (Table 2, "Configuration for Hiera variants"; FLOPs at 224)
| model | #Channels | #Blocks | #Heads | FLOPs | Params |
|---|---|---|---|---|---|
| Hiera-B+ | [112-224-448-896] | [2-3-16-3] | [2-4-8-16] | 13G | 70M |
| Hiera-L | [144-288-576-1152] | [2-6-36-4] | [2-4-8-16] | 40G | 214M |
- B+ total blocks 24, L total 48 (our arithmetic).
- Flag: Table 4 (K400 video) lists Hiera-B+ at 69M; Table 2 and Table 8 say 70M. sam2.md cites 70M.

### 8. Results
IN-1K top-1 (Table 8): Hiera-B+ 85.2; Hiera-L 86.1 (Sec. 6.1 text).

## 10. Outline feedback
1. ch-sam-arch: "ViT-H/16" is right for SAM, but the chain is: ViT trains H/14 (ViT Table 2) -> MAE pretrains H/14 (MAE Table 3 caption) -> ViTDet interpolates patch filters 14 -> 16 after pretraining (ViTDet App. A.2). Say "ViT-H, patch 16 after ViTDet-style interpolation" or cite SAM's own wording.
2. ViT-H width/depth/heads/MLP = 1280 / 32 / 16 / 5120 from ViT Table 1 (resolves the "uncertain" in sam-model.md row "width / depth / heads").
3. Window 14x14 = MAE pretraining map size (224/16 = 14 per ViTDet footnote, with patch 16). 64 is not divisible by 14 (padding; see sam-model.md).
4. Params: show "632M" (ViT/MAE) and "636M" (SAM Fig. 13) as two labelled numbers, not one.
5. ch-sam2: Hiera stage strides 4/8/16/32 are our reading; Hiera prints stage resolutions at 224.

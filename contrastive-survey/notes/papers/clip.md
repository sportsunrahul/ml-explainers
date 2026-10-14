# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## 1. Identity
- arXiv 2103.00020, version read: v1 (only version; abs.html: Fri 26 Feb 2021).
- Authors: Alec Radford ... Ilya Sutskever (12 authors; OpenAI).
- Venue: not stated in the arXiv PDF (template is ICML-style; do not cite a venue from this source).
- Repo: not given in sources folder.

## 2. Core problem
Vision models trained on a fixed label set need new labeled data for every new concept. Raw text paired with
web images is a much broader, cheaper supervision source. The question is how to learn from it efficiently
enough to scale.

## 3. Prior methods and why that hurt (per the paper)
- Caption prediction (VirTex-like transformer LM): "learns to recognize ImageNet classes three times slower than
  a much simpler baseline that predicts a bag-of-words encoding" (Sec. 2.3, Fig. 2).
- Predicting the exact words is hard "due to the wide variety of descriptions, comments, and related text".
- Earlier datasets small (MS-COCO, Visual Genome ~100,000 photos) or noisy (YFCC100M -> 15 million after
  English filtering) (Sec. 2.2).

## 4. Key idea
Predict only WHICH text in the batch goes with which image: an N x N similarity matrix, symmetric cross
entropy over rows and columns, learnable temperature. Text then acts as a classifier at test time (zero-shot).

## 5. Mechanism
1. Batch of n (image, text) pairs; batch size 32,768.
2. Image encoder (ResNet with attention pooling, or ViT) -> I_f [n, d_i]; text transformer -> T_f [n, d_t]
   (activations at [EOS], layer-normed).
3. Linear projections W_i, W_t to d_e (no non-linear head, Sec. 2.3), then L2-normalize.
4. logits = I_e T_e^T * exp(t): n x n scaled cosine similarities.
5. labels = arange(n): the diagonal is positive, the n^2 - n off-diagonal pairings are negatives.
6. Cross entropy along axis 0 and along axis 1, average. One softmax per column and one per row.
7. Zero-shot: embed class prompts ("a photo of a {label}.") and pick the most similar text.

## 6. Equations
No numbered loss equation in the paper. The loss is defined by Figure 3 pseudocode (src/pseudocode.pdf),
transcribed exactly:
```
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t)/2
```
Paper wording (Sec. 2.3): "CLIP is trained to predict which of the N x N possible (image, text) pairings across
a batch actually occurred ... maximize the cosine similarity of the image and text embeddings of the N real
pairs ... while minimizing the cosine similarity of the embeddings of the N^2 - N incorrect pairings. We
optimize a symmetric cross entropy loss over these similarity scores." The paper names its ancestors: multi-class
N-pair loss (Sohn 2016), InfoNCE (Oord 2018), ConVIRT (Zhang 2020).
Gloss: exp(t) is the inverse temperature (logit scale); t is log-parameterized; positions graded = every cell,
diagonal as target, each row and column normalized separately (our reading of axis=0/axis=1).

## 7. Config
| item | value | loc |
|---|---|---|
| Data | WIT: 400 million (image, text) pairs; 500,000 queries; up to 20,000 pairs per query | Sec. 2.2 |
| Batch | 32,768 | Sec. 2.5; Table 18 |
| Epochs | 32 | Sec. 2.5; Table 18 |
| Optimizer | Adam, decoupled weight decay 0.2, cosine lr; warm-up 2000 iters; beta2 0.999 (ResNet) / 0.98 (ViT) | Sec. 2.5; Table 18 |
| Temperature | init "equivalent of 0.07"; clipped so logits not scaled by more than 100 (max temperature 100.0) | Sec. 2.5; Table 18 |
| Augmentation | random square crop from resized images only | Sec. 2.3 |
| Projection | linear only | Sec. 2.3 |
| Text encoder | 63M-param, 12-layer, 512-wide, 8 heads; BPE vocab 49,152 (text) vs 49408 (Table 18); max len 76 | Sec. 2.4; Table 18 |
| Models | RN50, RN101, RN50x4, RN50x16, RN50x64; ViT-B/32, B/16, L/14 (+ L/14@336px, one extra epoch at 336) | Sec. 2.5 |
| Embedding dim d_e | RN50 1024; ViT-B 512; ViT-L/14 768 | Tables 19-20 |
| Compute | RN50x64: 18 days on 592 V100 GPUs; largest ViT: 12 days on 256 V100 GPUs | Sec. 2.5 |
| Memory tricks | mixed precision, gradient checkpointing, half-precision Adam stats, sharded similarity computation | Sec. 2.5 |

## 8. Results
- Zero-shot ImageNet top-1, best model (ViT-L/14@336px): 76.2% (Table 1 vs Visual N-Grams 11.5%; Table 11 last col).
  Top-5 95% (Sec. 3.1.3). Matches the original ResNet-50 without its 1.28 million labeled images.
- Zero-shot ImageNet, CLIP RN50: 59.6 (Table 11, RN50 row, ImageNet column).
- Linear probe ImageNet, L/14-336px: 85.4 (Table 10, last column).
- Fig. 2 (efficiency, zero-shot ImageNet vs images processed, up to 400M): transformer LM learns "3x slower"
  than bag-of-words prediction; swapping to the contrastive objective is "another 4x" (caption; plot labels
  "3X efficiency", "4X efficiency"). Curve values not printed.
- Prompt engineering + ensembling: "improve ImageNet accuracy by almost 5%" (Sec. 3.1.4).

## 9. Limitations stated (Sec. 6)
Weak on fine-grained (cars, flowers, aircraft), counting, truly OOD data (88% on MNIST, beaten by logistic
regression on pixels); limited to concepts in the given zero-shot classifier; needs big batches and compute.
Later: SigLIP replaces the batch-wide softmax with per-pair sigmoid (memory, small batch).

## 10. Outline feedback
- The outline is right that the loss is a transcription of Fig. 3 pseudocode (no equation number).
- Text says vocab 49,152, Table 18 says 49408: do not print vocab size, or cite one location.
- Use 76.2 (L/14@336px) and 59.6 (RN50) as two separate zero-shot claims; 85.4 is linear probe, keep separate.

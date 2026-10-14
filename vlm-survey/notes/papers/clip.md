# CLIP (2103.00020v1) -- extraction for ch-shared-space (+ ch-patches 224 vs 336 fact)

## Mechanism (plain words)
Given a batch of N (image, text) pairs, predict which of the N x N possible pairings actually occurred.
Both encoders map to a shared embedding space (linear projection only, no non-linear head), embeddings
are L2-normalised, cosine similarities are scaled by a learned temperature, and a symmetric cross
entropy is applied over rows and columns. N real pairs are positives, N^2 - N are negatives.
Trained from scratch (no ImageNet init, no pretrained text encoder). Only augmentation: random square crop.
Lineage per paper: multi-class N-pair loss (Sohn 2016), popularised as InfoNCE (Oord 2018), adapted to
(text, image) in medical imaging by ConVIRT (Zhang 2020).

## Fig. 3 pseudocode (verbatim; Fig. "pseudocode", caption "Numpy-like pseudocode for the core of an implementation of CLIP."; unnumbered, no equation number)
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
Gloss: I_f/T_f raw encoder features; W_i/W_t linear projections to d_e; logits[i,j] = cos(image i, text j) * exp(t);
labels = diagonal is the correct answer; loss_i and loss_t are the column-wise and row-wise cross entropy, averaged.
NOTE: the pseudocode multiplies by exp(t), i.e. the paper's "tau" is 1/exp(t) (logit scale = exp(t)); the prose
(Sec. 2.5 zero-shot) says cosine similarity is "scaled by a temperature parameter tau". Pseudocode never shows the clip.
Acknowledgments thank Ishaan Gulrajani "for catching an error in the pseudocode" (v1 is the version read).

## Temperature (Sec. 2.5 Training, para 1; Sec. 2.3 simplifications)
- Learnable, "directly optimized during training as a log-parameterized multiplicative scalar to avoid turning as a hyper-parameter" (Sec. 2.3).
- "initialized to the equivalent of 0.07 from (Wu 2018)" and "clipped to prevent scaling the logits by more than 100 which we found necessary to prevent training instability" (Sec. 2.5).
  So tau init = 0.07 (logit scale 1/0.07 ~ 14.3), max logit scale 100 (tau_min = 0.01).
- Minibatch size 32,768. Adam with decoupled weight decay, cosine LR, 32 epochs, mixed precision.
- Sec. 3.1.1: pre-training step viewed as a proxy dataset "which contains 1 example per class and has 32,768 total classes".
- Similarity computation sharded: each GPU computes only the subset of pairwise similarities needed for its local batch.

## Dataset (Sec. 2.2)
WIT (WebImageText): 400 million (image, text) pairs; 500,000 queries (words >=100 times in English Wikipedia, PMI bigrams,
Wikipedia article names above a search volume, WordNet synsets); up to 20,000 pairs per query (approx class balanced);
"similar total word count as the WebText dataset used to train GPT-2".

## Zero-shot classification (Sec. 3.1.2)
Class names embedded by the text encoder, cosine similarity with image embedding, scaled by temperature, softmax.
Equivalent to multinomial logistic regression with L2-normalised inputs and weights, no bias, temperature scaling; the
text encoder is a "hypernetwork" generating the classifier weights. Classifier cached once and reused.

## Prompt template (Sec. 3.1.4)
- Template "A photo of a {label}." (paper text: `A photo of a \{label\}.`), a "good default"; reason: pre-training text is
  usually a full sentence, rarely a single word.
- Gain: "just using this prompt improves accuracy on ImageNet by 1.3%".
- Ensembling 80 context prompts on ImageNet: additional +3.5% over the single default prompt (ensemble built in
  embedding space, averaged text embeddings, cached, so same cost as one classifier).
- Together: "almost 5%" on ImageNet; across 36 datasets "almost 5 points on average" (Fig. 4 caption), "similar to
  the gain from using 4 times more compute with the baseline zero-shot method".
- Polysemy example: ImageNet has construction cranes and flying cranes; Pets "boxer".
- Task-specific example: "A photo of a {label}, a type of pet." for Oxford-IIIT Pets.

## Zero-shot ImageNet (Sec. 3.1.2 + Table 1 + robustness Table)
- Table 1: CLIP 76.2 on ImageNet (aYahoo 98.4, SUN 58.5) vs Visual N-Grams 11.5 on ImageNet (72.4, 23.0).
- "Unless otherwise specified, all results reported in this paper as CLIP use this model [ViT-L/14@336px]".
  So 76.2% zero-shot top-1 is ViT-L/14@336px. Top-5 = 95%, "matching Inception-V4".
- Comparison stated in text: "matches the performance of the original ResNet-50 despite using none of the 1.28 million
  crowd-labeled training examples". Abstract: "we match the accuracy of the original ResNet-50 on ImageNet zero-shot".
  The paper does NOT print the ResNet-50 supervised top-1 number in the LaTeX text I could find (see wrong/unconfirmed).
- Linear probe CLIP on ImageNet 85.4 (robustness table, "Linear Probe CLIP"); Zero-shot CLIP 76.2; NS EfficientNet-L2 88.3.
- Fig. 5: zero-shot CLIP beats a fully supervised linear classifier on ResNet-50 features on 16 of 27 datasets, including ImageNet.

## 224 vs 336 (Sec. 2.5; Table 19 appendix)
- ViT-L/14 trained at 224 px; "we also pre-train at a higher 336 pixel resolution for one additional epoch to boost
  performance similar to FixRes. We denote this model as ViT-L/14@336px." (Sec. 2.5)
- Appendix hyperparameter table: ViT-L/14 224 px, LR 4e-4; ViT-L/14-336px 336 px, LR 2e-5; both embed dim 768, vision 24 layers
  width 1024, 16 heads, text 12 layers width 768 12 heads.
- ViT-B/32, B/16, L/14 use 224; the 336 variant is the ViT-L/14 fine-tuned at 336 (appendix A.3 text).
- Token count implied (computed, not in paper): 224/14 = 16 -> 256 patches; 336/14 = 24 -> 576 patches.
- Compute: largest ViT (L/14) 12 days on 256 V100; RN50x64 18 days on 592 V100.

## Outline assertions I could not confirm or that are wrong
- "zero-shot ImageNet ViT-L/14@336 (76.2%) vs ResNet-50 supervised": 76.2% confirmed for ViT-L/14@336px (default "CLIP").
  The paper text gives no numeric supervised ResNet-50 top-1; it only says zero-shot CLIP "matches" it. Do not display a
  ResNet-50 number attributed to CLIP; cite as "matches the original ResNet-50" (the commonly cited value ~76.1 is not in the source).
- "no numbered eq" for CLIP: confirmed (Fig. 3 pseudocode only).
- Temperature: the pseudocode uses exp(t) as a multiplier (logit scale), prose calls it tau; init "equivalent of 0.07",
  clip = logits scaled by no more than 100. Outline "max logit scale 100" matches.
- Prompt gain: 1.3% for the single template on ImageNet; "prompt engineering and ensembling" total is almost 5% (80 prompts add 3.5%).

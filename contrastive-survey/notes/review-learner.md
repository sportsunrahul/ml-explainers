# Learner review: "teach me the concept"

Reviewer stance: I know a CNN, softmax, cross-entropy and SGD. I know nothing about contrastive or
self-supervised learning. I read parts/ORDER in order, as stripped prose plus figure controls.
I did not edit any page file. I re-checked every worked example by hand; all the arithmetic is correct.

Overall: ch-compare, ch-infonce (first half), ch-sphere and ch-supcon teach well. The lesson breaks down
in three ways. (a) The same letters (N, M, K, t, m, tau) change meaning from chapter to chapter.
(b) Some terms are used chapters before they are defined. (c) The "what came next" sections at the
end of ch-infonce, ch-moco, ch-simclr, ch-negatives, ch-beyond and ch-clip turn into paper digests.

---

## Part 1. The ten changes that would most improve learning

### 1. ch-negatives: tau^+ / tau^- (class prior) next to tau (temperature)
- Where: "the papers call this probability the class prior \tau^+", then eq-debias, which has
  1/\tau^-, \tau^+ and e^{-1/\tau} in one display. The gloss also says Chuang writes the temperature as t.
- Problem: a learner reads tau as "temperature" for five chapters. Here three taus with two meanings
  share one equation. This is the worst notation trap on the page.
- Fix: in page notation, write the class prior as \pi^+ and \pi^- = 1 - \pi^+. Add one sentence:
  "The paper writes this prior as \tau^+. It is not a temperature; we write \pi^+ to keep tau for
  temperature." Apply the same change to eq-hard's gloss and to the "In practice" aside.

### 2. Define "linear probe" and "top-1" at first use (ch-infonce)
- Where: ch-infonce, CPC flow step 8, says "a linear classifier on ImageNet labels reaches 48.7% top-1
  (linear probe, ResNet-v2-101)". Then "Where this came from" uses "linear probe" three times.
  The definition only arrives in ch-moco, "Where this came from": "A linear probe trains one classifier
  layer on frozen features." "top-1" is never defined anywhere.
- Problem: almost every number on the page is a "linear-probe top-1" number. The reader cannot judge
  any of them until chapter 4.
- Fix: put this into ch-infonce at the first number: "How do you score an encoder that never saw a
  label? Freeze it, train one linear layer on top with the labels, and report how often its top guess
  is right (top-1 accuracy). This is a linear probe; every accuracy on this page uses it unless it says
  otherwise." Remove the late definition in ch-moco, or turn it into a back-reference.

### 3. ch-infonce: the CPC-notation equation comes before the reader knows CPC
- Where: "The same idea as an equation" shows eq-infonce with c_t, x_{t+k}, f_k, W_k and z_{t+k}.
  The reader meets CPC's patches, context network and k-steps-ahead prediction only later, in "Inside CPC".
- Problem: an equation in an unfamiliar setting arrives before its motivation. The page's own equation
  (eq-infonce-ours) is the one the reader needs, and it comes second.
- Fix: show eq-infonce-ours first, with its gloss and the worked example. Move eq-infonce (CPC form) into
  "Inside CPC", step 6, where c_t and x_{t+k} are drawn. Introduce it as: "This is the same loss, written
  in CPC's own notation."

### 4. One symbol, one meaning: N, M, K and t
- N: lineup size in ch-infonce ("N candidates: one partner and N-1 negatives"). The number of photos in
  a batch in ch-moco (N = 256), ch-simclr (2N views, 2(N-1) negatives) and ch-step. The number of
  negatives in eq-debias. The number of images in eq-dimcov. The number of photo-caption pairs in ch-clip.
- M: the number of negatives in eq-limit (ch-sphere). The number of positives in eq-debias (ch-negatives).
- K: the queue length in ch-moco. The number of prototypes in ch-beyond.
- t: the CPC time index; Tschannen's target ("a target t of 2 or 4"); the uniformity bump width in
  ch-sphere; Chuang's temperature; CLIP's logit scale; SigLIP's "b" for the per-device batch.
- Fix: you cannot rename the quoted equations. Add a short "Careful" line at each switch, and add one
  shared notation box in ch-infonce. Minimum set:
  - ch-simclr, Lesson 2: "Careful: N now counts photos, not candidates. Each row's lineup has 2N - 1
    candidates."
  - ch-sphere, after eq-limit: "M here is the number of negatives, the N - 1 of chapter 3."
  - ch-negatives, eq-debias gloss: "Here N counts negatives and M counts positives. In chapter 6, M
    counted negatives."
  - ch-beyond: "K here is the number of prototypes, not MoCo's queue length."

### 5. ch-clip, "Where it stands in 2025-26": a method zoo with eight entries
- Where: the eight bullets (ALIGN, OpenCLIP, MetaCLIP, SigLIP 2, Perception Encoder, Web-SSL,
  DINOv2/v3, DenseCL/DetCon). Each bullet has a long sentence (34 to 37 words) and a headline number.
- Problem: this reads like a related-work section. It brings in new jargon (self-distillation,
  captioning decoder, masked prediction, mIoU, ADE20k, VQA, power laws) and teaches nothing.
- Fix: reduce it to one lesson with three examples: "Since 2021 the loss has hardly changed; the gains
  came from data (ALIGN, MetaCLIP) and scale (OpenCLIP, Perception Encoder)." Put dense features in one
  sentence that points to ch-guide. Move the rest to closing, "Where to go deeper".

### 6. ch-beyond: the "third answer: asymmetry" teaches nothing, but other chapters depend on it
- Where: "BYOL ... and SimSiam ... use no negatives and no balancing rule. One branch gets an extra small
  network, the predictor; the other branch gets no gradient ... DINO ... centering and sharpening of the
  teacher's output. Why asymmetry works is a story of its own; the JEPA survey's collapse chapter tells it."
- Problem: ch-moco promises "adds a prediction head on the query side, as BYOL does (chapter 8)". The
  map and the guide recommend DINOv2/DINOv3 ("self-distillation"). But "teacher", "self-distillation",
  "centering" and "sharpening" are never defined. The JEPA survey is not linked, and this reader has not
  read it.
- Fix: add one short h3 step: what a predictor is, what stop-gradient is (brown means no gradient), and
  the one-sentence intuition. Define "self-distillation" as "a student network learns to match a
  slow-moving teacher copy of itself, as MoCo's key encoder does". Then link the JEPA page with a real
  URL. Cut VICReg's "\lambda = \mu = 25 and \nu = 1".

### 7. ch-simclr: the meanings of "embedding" and "representation" change
- Where: ch-compare defines "The network that turns an image into a vector is the encoder. Its output
  vector is the embedding." ch-simclr then splits encoder output h from head output z. The loss sees z,
  and you keep h. ch-beyond says "the 128-d embedding that the loss sees" against "the 2048-d
  representation".
- Problem: by ch-compare's definition, h is the embedding. By ch-simclr and ch-beyond, z is the
  embedding. The reader is not told that the word changed meaning.
- Fix: at the start of ch-simclr, Lesson 3, add: "From here on, the embedding is what the loss sees
  (z), and the representation is the encoder output you keep (h). Before this chapter there was no
  head, so the two were the same vector."

### 8. Terms used before they are defined
- ch-moco, MoCo v2: "A two-layer MLP head with a hidden size of 2048 replaced the single layer". The
  reader learns what the head is for only in ch-simclr. Add: "(a projection head, explained in
  chapter 5)".
- ch-moco, shuffle-BN aside: "the pretext accuracy climbs above 99.9%". "Pretext" is never defined.
  Write: "the accuracy on the training game (picking the partner) climbs above 99.9%".
- ch-moco, MoCo v3: "each term is scaled by 2\tau". It is unclear what is scaled or why. Cut it, or say
  "the loss is multiplied by 2\tau".
- ch-simclr, step 8, and ch-step: "no stop-gradient". The term is defined only in ch-beyond. Write
  "no part that is cut off from the gradient (a stop-gradient)".
- ch-map: the tree text names CPC, the memory bank, alignment, uniformity, DINOv2 and Web-SSL before
  the reader meets any of them. Add one sentence before the tree: "Do not worry about the names yet;
  each solid pill gets a chapter. Come back to this map after chapter 8."
- ch-sphere: "\triangleq" and "i.i.d." are not explained. Add a gloss line: "\triangleq means 'is
  defined as'; i.i.d. means drawn independently from the same distribution."
- ch-beyond: "negative eigenvalue" in Theorem 1. One clause helps: "a direction in which X shrinks
  vectors instead of growing them".

### 9. ch-simclr "What came next" and ch-moco "What changed next": results digests
- ch-moco: the MoCo v2/v3 section has three paragraphs of numbers (5.6 points, 0.2, 71.1, 74.3, 2.2,
  72.6 -> 69.7, 1.7, 76.7, 76.5). It has one real lesson: with a big batch, the queue is unnecessary,
  but the momentum encoder still helps. Keep that lesson and the ViT-instability lesson. Drop the
  number chain, or move it into a small table under a "Sourced" badge.
- ch-simclr: the InfoMin paragraph (a 42-word sentence about "InfoMin Aug." and a jigsaw branch) and
  the SimCLRv2 paragraph (selective kernels, distillation into a student, 1% and 10% labels) introduce
  new ideas in passing. Keep InfoMin's idea in two sentences, because it explains Lesson 1. Cut the
  rest, or move it to closing, "Where to go deeper".
- ch-negatives, "Where this came from": nine sentences, with three experimental setups and five
  numbers. Cut it to who did what. Move the numbers into the STL10 table caption, or remove them.
- ch-infonce, "Where this came from": CPC v2, CMC, Poole and Arora are listed with numbers. The Arora
  sentence ("in that paper \tau is the probability that two independently drawn classes are the same")
  adds a third meaning of tau only to warn about it. Cut the Arora sentence.

### 10. ch-step duplicates ch-simclr's flow figure, and the interlude has no check
- Where: ch-simclr "Inside SimCLR" (viz-simclrflow, nine stages) and ch-step (viz-step, ten stages)
  walk the same step. They give the same shapes, the same 8190 negatives and the same LARS learning
  rate.
- Problem: the reader walks the same pipeline twice in a row. That costs attention and teaches nothing
  new. ch-step also has no self-check and no opening question.
- Fix: keep one. Option A: remove viz-simclrflow from ch-simclr, and end Lesson 3 with "The interlude
  runs one full step." Option B: keep both, but make ch-step about the one new idea it has (each
  negative's gradient weight equals its softmax probability). Then add a self-check: "In one row at
  N = 4096, how many cells get a push, and which one gets a pull?"

---

## Part 2. Chapter by chapter (the rest)

### ch-compare (model chapter)
- Good. Paragraph "Press it and watch..." has 7 sentences. Split after "This failure is called collapse."
- "Where this came from": "A year earlier, Chopra, Hadsell and LeCun had trained a siamese network ..."
  is 33 words. Split at the colon.
- "Later work showed that, when the score is a dot product of two embeddings, it is the same objective
  as InfoNCE ..." (37 words). The reader has not met InfoNCE yet. Write: "Later work showed that
  N-pair is the same loss as InfoNCE, which chapter 3 builds."
- Symbol m is the margin here. ch-moco reuses m for Wu's sample count and for the momentum. ch-moco
  flags only the first clash. Add "(not the margin m of chapter 1)" at the first use of m as momentum.

### ch-map
- No opening question in the learner's voice. The current opening is fine; you could make it "How big
  is this field, and which branches does this page follow?" (it is nearly there).
- "The chart below asks the arXiv preprint server two questions ..." is 43 words. Split it into two
  sentences, one per question.
- "Chapters 4 and 5 open each recipe up." has no links. Link #ch-moco and #ch-simclr.
- "Dense: ... The decision guide at the end comes back to it." This is the only lane without a chapter.
  Say so: "This page does not teach it; the guide and chapter 10 point to it."
- The chapter is placed between the first lesson and InfoNCE. A learner who wants the mechanics loses
  momentum here. Consider moving it after ch-infonce, or mark it "optional on first read".

### ch-infonce
- The Tschannen claim card is dense with jargon: "RealNVP encoder (30 coupling layers)", "trained
  adversarially", "separable and MLP critics have 40k parameters", "bilinear critic". Keep Test 3 (same
  bound, different features), because it makes the point by itself. Cut or gloss the rest.
- "Their conclusion, in their words, is that maximizing tighter bounds ..." is 37 words. Split it after
  the quote.
- "Set the figure to hard negatives, N = 64 and \tau = 0.05: ..." is 44 words. Split it: "Set ... Now
  the partner's lost probability goes almost all to the closest negatives."
- "Inside CPC": "PixelCNN-style network", "ResNet-v2-101 ... without batch normalization", "third
  residual block ... mean-pooled". This is fine as figure detail, but gloss PixelCNN as "a network that
  only looks at rows above".
- The "Mutual information" paragraph is good. Add the word "estimate" once, so the reader does not
  think I is computed.

### ch-moco
- Map the vocabulary explicitly. Add after the query and key definitions: "The query is chapter 3's
  anchor. The keys are its candidates: one positive key and many negative keys."
- The momentum table and caption sit inside the figure. The prediction prompt is good.
- "Where this came from": "measured with a linear SVM on conv5 features". Write "with a different
  linear classifier on an earlier layer", or drop it.
- "In practice": "Treat m as a delay in steps, about 1/(1-m)" is good. Tie it to the worked example
  ("about 1000 steps at 0.999").
- There are two self-checks, which is good. The m = 1 check is a strong question.

### ch-simclr
- The opening sentence is 37 words and contains a semicolon chain. Write: "MoCo built a queue and a
  momentum encoder to get many negatives. SimCLR asks whether you need any of that. Its answer is no,
  if you get three plain choices right. The most important choice is how you make the two views."
- Color-strength table: "AutoAug" is undefined. Write "AutoAugment (a learned augmentation policy)",
  or drop the column.
- SimCLR Table 3: in "color or grey-scale ... random guess 80", a learner reads "random guess" on a
  yes/no task as 50. Rename the column "always guess the common answer". Add one line: grey-scale is
  applied 20% of the time, so always guessing "color" scores 80.
- "A nonlinear head ... beats a linear head by 3% and beats no head by more than 10%; and even with ..."
  is 52 words. Split it into two sentences. Write "points", not "%", for accuracy differences.
- In eq-ntxent the positive term is colored t-green. In eq-infonce-ours it is t-yellow. The page key
  says yellow = positive. Use yellow for the partner term; green can remain for agreement in figures.
- Step 9 (LARS, TPU v3 cores, 1.5 hours) is fine as figure text. It is not needed for the lesson.

### ch-step (interlude)
- See item 10. Also: "In our summary of LARS (You et al., 2017; ...), each layer's step is scaled ..."
  is 45 words. Split it.
- "In practice" (42 words, first sentence). Split it after "it is split".

### ch-sphere
- This is a strong chapter. eq-limit has no worked example. Add one: "Take \tau = 0.5, a partner
  cosine of 0.9, and negatives that all sit at cosine 0. The alignment term is -0.9 / 0.5 = -1.8. The
  uniformity term is \log e^{0} = 0. Move one negative to cosine 0.5 and the uniformity term rises above 0."
  Or say explicitly that eq-bridge is the worked step for it.
- The caption uses \mathcal L_\mathsf{align} (\alpha = 2) and \mathcal L_\mathsf{uniform} (t = 2) before
  the reader meets them. Move these into the prose after eq-uniform, or say "defined below".
- The HaoChen et al. paragraph (a 42-word sentence about the spectral contrastive loss) is a dangling
  method. Cut it, or move it to "Where to go deeper".
- The Wang and Liu quote is 42 words. Paraphrase it in one 15-word sentence, and keep the quote short.
- In the Wang and Isola claim card, "77.51% vs 73.76%" does not say which is which. Write "InfoNCE
  77.51% vs align + uniform 73.76%".

### ch-negatives
- The opening is a statement, not a question. Write: "InfoNCE pushes every negative away. Does it push
  them all equally, and what if some 'negatives' are not negatives?"
- Cross-paper confusion: the STL10 table shows SimCLR 80.15 -> debiased 84.26. Two paragraphs later,
  Chuang says debiasing improves accuracy "by 4.26%". 84.26 - 80.15 = 4.11. The two numbers come from
  different papers and runs, but the reader sees 84.26 and 4.26 and suspects a typo. Say "in Chuang's
  own run" next to the quote.
- eq-hard has no worked example. Add one with two negatives at s = 0.9 and 0.1 and \beta = 1. Show
  that the weights e^{\beta s} / mean shift mass to the near one. It mirrors the share example above.
- The paragraph "Here is why. Write s_{i,j} ..." has 7 sentences and derives a gradient inline. Split
  it before "Take the derivative".
- "Both at once": the paragraph after the table lists tinyImageNet, CIFAR100, CIFAR10 and MoCo-v2
  numbers. Keep one sentence: "The same order holds on other datasets, but the gain shrinks when there
  are many negatives."

### ch-beyond
- This is the longest chapter (about 2600 words), with three ideas and a zoo. Consider a split:
  8a "Clusters and redundancy" and 8b "Dimensional collapse". If you do not split it, cut "How the
  methods compare" to the table plus one sentence. The SwAV-baseline bookkeeping ("Barlow Twins prints
  71.8 for 'SwAV without multi-crop'; SwAV never prints that number ...") is reviewer-level detail.
- "Where this came from", second paragraph: iBOT, DINOv2, DINOv3 and Web-SSL with headline numbers.
  The reader has no frame for "masked-patch prediction", "teacher" or "ViT-g/14". Cut this to one
  sentence that points to item 6's new step.
- The Sinkhorn hand example is 8 sentences. Split it after "Both images prefer prototype 1."
- The committee analogy is good. It is the chapter's second analogy (after "bins"). This is fine, but
  do not add a third.

### ch-supcon
- This is a good chapter. One letter clash: "the softmax over every other view gives a probability to
  each candidate p", but p is then the positive index. Write "each candidate a".
- The triplet sentence is 48 words. Split it: "The supplement takes one positive and one negative. When the
  anchor is already much closer to its positive, SupCon behaves like a triplet loss on squared
  distances. The margin is 2\tau."

### ch-clip
- The opening sentence is 36 words. Split it into three, one per question.
- "The VLM survey's sigmoid chapter animates this ..." and "see the VLM survey ... the open-vocabulary
  detection survey" are not links. Add URLs, or cut them.
- "memory ... drops from |\mathcal B|^2 to b^2, where the paper's b here means the per-device batch, not
  the bias". Write B_d (our notation) instead of b, and say so.
- "Before you press anything" plus "Make a guess" is a good prediction prompt.

### ch-guide
- No prediction prompt, which is acceptable for a guide. The "dense" recipe sends the reader to "chapter
  10, 'Where it stands'". If item 5 cuts that list, retarget the link.
- "It appears as a loss you fine-tune with ..." is 43 words. Use three short sentences.

### ch-closing
- The Summary paragraph has 7 sentences, most of them 35 to 41 words. As a recap this is acceptable,
  but split it into a 6-item list, one item per act.
- "For BYOL, SimSiam and DINO, read the JEPA survey." Give the URL.

---

## Part 3. Cross-chapter consistency table

| Symbol or word | Meanings found | Where |
|---|---|---|
| tau | temperature; class prior tau^+/tau^-; Arora's same-class probability | ch3-ch10; ch-negatives; ch-infonce |
| t | CPC time index; Tschannen target; uniformity width (= 1/(2 tau)); Chuang temperature; CLIP scale (= 1/tau) | ch-infonce; ch-sphere; ch-negatives; ch-clip |
| m | margin; Wu's sample count; momentum | ch-compare; ch-moco (flagged once) |
| N | lineup size; photos per batch; negatives; images; pairs | ch-infonce; ch-moco/simclr/step; ch-negatives; ch-beyond; ch-clip |
| M | negatives; positives | ch-sphere; ch-negatives |
| K | queue length; prototypes | ch-moco; ch-beyond |
| b | SigLIP bias; per-device batch | ch-clip (flagged inline) |
| embedding | encoder output; head output z | ch-compare; ch-simclr/ch-beyond |
| anchor/positive vs query/key | the same roles, two vocabularies | ch-infonce vs ch-moco |
| positive color | yellow (key); green in eq-ntxent | ch-infonce vs ch-simclr |

Cross-references I checked: every #ch-... link lands on a chapter that does what the link says, with
three exceptions. (1) ch-moco "prediction head ... as BYOL does (chapter 8)": ch-beyond gives BYOL one
sentence. (2) ch-map "Dense ... the decision guide comes back to it": the guide forwards to ch-clip.
(3) ch-guide "chapter 10, 'Where it stands'": this is fine now, but it breaks if item 5 is applied.
The chapter numbers in prose match the TOC (ch-step is unnumbered, as intended).

Teaching-device checklist (missing items only): ch-step has no self-check and no opening question.
ch-negatives has no opening question. eq-limit, eq-hard, eq-crosscorr and eq-dimcov have no worked
example right after them. Every chapter except ch-closing has a recap. Every chapter except ch-guide and
ch-closing has a prediction prompt.

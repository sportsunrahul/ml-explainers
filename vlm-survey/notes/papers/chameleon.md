# Chameleon (2405.09818v2) -- chapter 11 ch-early-fusion

Source: sources/chameleon/src/paper.tex (+ tables/complete_win_rates.tex, task_fulfillment_breakdown.tex).

## Mechanism in plain words
Every modality becomes discrete tokens in ONE vocabulary. Images go through a VQ image tokenizer (based on
Make-A-Scene) to 1024 codebook ids; text goes through a BPE tokenizer. Both are concatenated into one sequence
and a single LLaMa-2-style decoder-only transformer, trained from scratch end to end with a plain next-token
objective, reads and writes any interleaving (text-in/text-out, image-in, image-out, mixed). No separate
image encoder, no separate image decoder, no connector ("early fusion": all modalities in a shared space from
the first layer). Image tokens are 8192 entries inside a 65,536-entry vocabulary, so 1/8 of the vocabulary is
image codes. For text-image pairs, 50% of the time the image comes before the text (captioning direction).
Tokenizer weakness (paper): reconstruction of images with lots of text is poor, which "upper bounds" OCR-heavy
tasks.

## Tokenizers
- Image tokenizer: "encodes a 512x512 image into 1024 discrete tokens from a codebook of size 8192"; trained on
  licensed images only; images with faces upsampled 2x. (Sec. 2.1 paragraph is rendered text. The
  tokenizer-comparison figure with captions CM3 256x256 / 256 tokens / vocab 1024, CM3Leon 256x256 / 1024 /
  8192, Ours 512x512 / 1024 / 8192 is inside a \begin{comment} block, so it may not appear in the PDF.)
  So 512/1024 tokens = each token covers a 16x16 pixel patch region (32x32 grid; my arithmetic, paper does not
  state 32x32 in the text I read).
- Text tokenizer: BPE via sentencepiece, "vocabulary size of 65,536, which includes the 8192 image codebook
  tokens".

## Models, data, compute
- Sizes: 7B and 34B (Table llama_models: LLaMa-2-style, context 4k; 34B uses GQA, 7B does not).
- Data: stage 1 = first 80% of training, stage 2 = last 20% (stage-1 data weight lowered 50%, higher-quality data
  added, filtered instruction-tune train sets). Stage-1 data: text-only 2.9 trillion tokens; text-image 1.4
  billion pairs = 1.5 trillion tokens; interleaved 400 billion tokens.
- Training tokens: Table llama_models lists "Tokens 4.4T, Epochs 2.1" for both sizes; text: "2.1 epochs over our
  full training dataset for a total of 9.2 trillion tokens seen during training." Figure 1 caption: "~10T
  tokens". Intro: 34B trained "on 5x the number of tokens as Llama-2" (9.2T vs 2.0T = 4.6x; loosely "5x").
- Batch: 7B 2^23 (~8M) tokens; 34B 3 x 2^22 (~12M) tokens.
- Compute: 7B 1024 concurrent GPUs, 856,481 GPU hours; 34B 3072 GPUs, 4,282,407 GPU hours (A100 80GB).
- Optimiser: AdamW b1=0.9 b2=0.95 eps=1e-5, 4000 warmup steps, exponential decay to 0, wd 0.1, clip 1.0,
  LR 1.0e-4 (LLaMa-2 used 3.0e-4 for 7B, 1.5e-4 for 34B).

## Stability (Sec. 2.3)
Problem statement: "It was challenging to maintain stable training when scaling the models above 8B parameters
and 1T tokens, with instabilities often only arising very late in training." Divergences appeared "after as
much as even 20-30% of training progress".
Diagnosis: slow norm growth in mid-to-late training. Softmax is translation invariant, softmax(z) = softmax(z+c)
(unnumbered, Sec. 2.3 Architecture). Because all weights are shared across modalities of very different
entropy, "each modality will try to compete with the other by increasing its norms slightly"; harmless early,
diverges once norms leave the representable range of bf16. Evidence it is the multimodal mix: an ablation with
image generation disabled did not diverge (Fig. no_imagegen). This is "logit drift" in unimodal literature.
Early warning: uncontrolled growth of the last-layer output norms "is strongly correlated with predicting
future loss divergence" (Fig. output_norms).
Fixes and what failed without them:
1. QK-Norm: layer norm on the query and key vectors inside attention, bounding the input norm of the attention
   softmax. Without it, Chameleon-7B "diverges after approximately 20% of a training epoch" (Fig. qknorm).
   "QK-norm is essential in both cases" (7B and 34B).
2. Dropout (0.1 after attention and feed-forward layers): needed for 7B in addition to QK-Norm (Fig.
   dropout, 7B with vs without). Not used for 34B (dropout 0.0): with norm reordering it "did not work well in
   combination with dropout"; and for 34B "using dropout does not fix divergences, both with and without
   norm-reordering" (Fig. 34b_dropout). Retroactively 7B can also train stably without dropout given
   norm reordering.
3. Norm reordering (Swin-style normalisation), needed for 34B; QK-Norm + dropout "was not enough to
   stabilize" 34B. Equations, unnumbered, Sec. 2.3 Architecture:
     Chameleon-34B:  h = x + attention_norm(attention(x));   output = h + ffn_norm(feed_forward(h))
     Llama2:         h = x + attention(attention_norm(x));    output = h + feed_forward(ffn_norm(h))
   x = block input, h = hidden vector after self-attention, attention_norm/ffn_norm = RMSNorm. Chameleon
   applies the norm to the OUTPUT of each sublayer (post-norm on the branch, before the residual add); Llama-2
   normalises the sublayer INPUT (pre-norm). Motive: "bounds the norm growth of the feedforward block, which
   can become additionally problematic given the multiplicative nature of the SwiGLU activation". No
   perplexity difference from reordering until the LLaMa-2 parameterisation diverges.
4. z-loss: QK-Norm fixes the inner softmaxes but "does not solve the problem of logit shift in the final
   softmax". Add 1e-5 * log^2 Z to the loss, Z = sum_i exp(x_i) the softmax partition function (unnumbered,
   Optimization paragraph; sigma(x)_i = e^{x_i}/Z). "For Chameleon-7B it was important to use both dropout
   and z-loss to achieve stability, while Chameleon-34B only required z-loss."
Summary table (Table llama_models): 7B: dropout 0.1, z-loss 1e-5, QK-norm yes; 34B: dropout 0.0, z-loss 1e-5,
QK-norm yes (+ norm reordering, in the text).
Note: architecture otherwise RMSNorm, SwiGLU, RoPE as LLaMa-2.

## Human evaluation (Sec. 4, "holistic" mixed-modal generation)
- Prompts: 1,048 crowdsourced prompts, 441 (42.1%) mixed-modal input, 607 (57.9%) text-only; all expect
  mixed-modal output; 12 task categories; each question answered by 3 annotators, majority vote.
- Compared: Chameleon-34B vs Gemini Pro and GPT-4V (both text-only output), plus enhanced baselines Gemini+ and
  GPT-4V+ where the model writes image captions in <caption> tags and DALL-E 3 renders them into the response.
- Relative metric: annotators choose first / second / about the same; win rate = wins + 0.5 x ties (footnote).
  Table complete_win_rates counts (wins / ties / loses / win rate):
    vs Gemini+:  435 / 362 / 251 / 58.8% overall (mixed-modal prompts 60.4%, text-only 57.6%)
    vs GPT-4V+:  375 / 331 / 342 / 51.6%
    vs Gemini:   561 / 327 / 160 / 69.1%
    vs GPT-4V:   482 / 329 / 237 / 61.7%
  Text: vs Gemini+ 41.5% win, 34.5% tie, 24.0% loss; vs GPT-4V+ 35.8 / 31.6 / 32.6; vs Gemini 53.5 / 31.2 / 15.3;
  vs GPT-4V 46.0 / 31.4 / 22.6. Abstract-level claim: "60.4% preference rate against Gemini-Pro and 51.6%
  against GPT-4V" in pairwise comparisons.
- Absolute metric (task fulfillment, 3 annotators): fully fulfils: Chameleon 55.2%, Gemini+ 37.6%, GPT-4V+
  44.7%; original Gemini 17.6%, GPT-4V 23.1%. Partially fulfilled: Chameleon 36.7%, Gemini+ 56.5%, GPT-4V+
  46.8% (in commented-out text).
- Weak categories for Chameleon: Story vs GPT-4V+ 18.3% win rate; Reasoning and Identification weakest in
  absolute fulfillment.
- Caveats the paper states: prompts crowdsourced not from real users; relative evaluation has >10% of pairs with
  no three-way agreement (counted as ties).
Other headline numbers (Sec. 1): Chameleon-34B beats Flamingo, IDEFICS, Llava-1.5 on captioning/VQA; MMLU 65.8
vs Mixtral 8x7B / Gemini-Pro 70.6 / 71.8.

## Outline assertions I could not confirm or that are wrong
- Outline "tokens trained (~10T?)": Figure 1 says ~10T; the training-details text says 9.2 trillion tokens seen
  (2.1 epochs); the architecture table says "Tokens 4.4T" (dataset size per epoch). Use 9.2T, mention "~10T"
  only as the figure caption's rounding. Note the stage-1 component sums (2.9T + 1.5T + 0.4T = 4.8T) do not
  equal the 4.4T table figure; the paper does not reconcile these (stage 2 reweights).
- Human-eval headline: the abstract-level text says Chameleon-34B "achieving a 60.4% preference rate against
  Gemini-Pro", but the paper's own win-rate table gives 58.8% overall vs Gemini+ (60.4% is the mixed-modal
  prompt subset) and 41.5 + 0.5 x 34.5 = 58.75. 51.6% vs GPT-4V+ is consistent. Also the baselines are the
  "+" versions (DALL-E 3 images added) not raw Gemini/GPT-4V; vs the raw models the win rates are 69.1% and
  61.7%.
- Outline "norm reordering" + "output-norm divergence": the norm reordering is only needed for 34B; 7B used
  dropout. Output-norm growth is a diagnostic signal, not a fix.

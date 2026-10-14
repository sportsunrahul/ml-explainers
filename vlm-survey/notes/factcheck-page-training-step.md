# Fact-check: "One training step, end to end" (section ch-training-step)

Scope: parts/ch-training-step.html (all prose, figure panels, caption, both equations) and parts/viz-pipeline.js (canvas
text, READ readouts, toy definitions). Every data-claim span (34 ids) was checked against the LaTeX in sources/*/src/,
with PDF renders or pdftotext where layout mattered (LLaVA Table 2 rendered from paper.pdf p.5; LLaVA and Flamingo
equation numbers from the PDFs). Toy numbers were recomputed by running the figure's own definitions (viz-pipeline.js
l.10-30) in node. The built explainer.html ships these parts verbatim (whitespace-normalised compare; math pre-rendered).
No page or claims file was edited.

Headline: all 34 claim values are right, 33 of 34 locs are right (item 25 has a half-wrong appendix pointer), and
both equations match the LaTeX symbol by symbol. The defects are in the prose around them. Three statements are false: every position has a target, padding is read as context,
and the padded batch is a tensor of token ids. Three mislead: the Eq. 3 gloss redefines L, the Qwen2-VL packing
sentence is about the ViT, and the pipeline joins LLaVA-1.5's encoder to LLaVA v1's schedules. Three labels are
missing: "at real scale" on toy percentages, the <image> placeholder/splice as implementation practice, and the
page's own reason for Flamingo's round-robin gap. The rest are nits.

## Problems

1. **wrong** (minor) - "every position has a target" vs positions followed by a visual token
   - Page: ch-training-step.html l.81 (6a panel): "each position is asked to predict the token that follows. Every
     position therefore has a target." Also l.129: "So every position comes with a target, the next token along."
     viz-pipeline.js l.492 (6a readout): "Each position of example A is paired with the next token as its target."
     And l.130: "the visual tokens have no target at all, since they are continuous vectors with no vocabulary entry
     to predict."
   - Source: LLaVA Eq. 3 factors are over "the current prediction token x_i" (approach.tex l.169-173); a visual
     token is H_v = W . Z_v, a vector (Eq. 1, l.111), so it can never be a target. Recomputed from the toy: in each
     example 9 positions are followed by a visual token (the one before <image> plus 8 inside the block), i.e. 27 of
     the 118 real positions have no target; drawTargets draws them as blank blue chips (viz-pipeline.js l.398).
     Conversely, in example A the <image> is followed directly by ### (l.14), so the last visual position (index
     17) predicts a graded ### and the figure draws a red target under a blue chip (l.395). So "every position has
     a target" is false, and "the visual tokens have no target" is true only of visual tokens as targets.
   - Replace with:
     l.81: "Every position followed by a word therefore has a target; a position followed by a visual token has none,
     since a vector is not in the vocabulary."
     l.129: "So every position followed by a word comes with a target, the next token along; positions followed by a
     visual token have none."
     l.130: "the visual tokens are never targets, since they are continuous vectors with no vocabulary entry to
     predict (the last visual position in example A is graded only because the token it predicts is the ### after
     the image)."
     JS l.492: "Each position of example A is paired with the next token as its target (none where the next token is
     visual). Nothing has been masked yet."

2. **wrong** (minor) - 6b panel says padding is read as context
   - Page: ch-training-step.html l.85 (6b panel): "Everything else is still read as context but gets no loss: the
     system message, the human's words, the role labels, the visual tokens and the padding."
   - Source: contradicts the page's own 5a (l.69: "an attention mask stops real tokens from looking at the
     padding") and viz-pipeline.js l.321 ("Pad positions are masked: real tokens never attend to them"). Padding is
     common practice (caption l.96), not from a paper.
   - Replace with: "Nothing else is ever a graded target: the system message, the human's words, the role labels
     and the visual tokens are read as context only, and the padding is not read at all (the pad mask hides it)."

3. **wrong** (minor) - padded batch described as token ids
   - Page: viz-pipeline.js l.489 (5a readout): "Padded batch: a 3 x ${LMAX} tensor of token ids, plus a mask saying
     which cells are real." (renders "a 3 x 50 tensor of token ids")
   - Source: the rows are drawn after splicing (9 visual cells each), and the page's own stage 4 says "The result is
     one sequence of vectors per example" (ch-training-step.html l.65). Visual tokens have no ids: H_v "have the
     same dimensionality as the word embedding space" (LLaVA approach.tex l.104-106).
   - Replace with: "Padded batch: a 3 x ${LMAX} grid of embedding vectors (visual and word alike), plus a mask
     saying which cells are real."

4. **misleading** - Eq. 3 gloss redefines L
   - Page: ch-training-step.html l.138: "x_i: the i-th graded token, one of the red targets in stage 6b; L is how
     many there are, and the product runs over only those."
   - Source: LLaVA approach.tex l.165 (PDF p.5, just before Eq. 3): "Specifically, for a sequence of length L, we
     compute the probability of the target answers X_a by:"; x_i is "the current prediction token" (l.173). The
     page's reading (only prediction tokens contribute) is the coherent one, but the paper defines L as the sequence
     length, and the gloss presents its own reading as the paper's. The same display also puts the page-added loss
     \mathcal{L} (line 2, labelled in the eqno) next to L used as a count.
   - Replace with: "x_i: the i-th token; the paper colors it as a prediction token and introduces the equation 'for
     a sequence of length L'. Read with Table 2, only the prediction tokens (the red targets in stage 6b) contribute
     factors, so in effect the product runs over those." Optional: write the second line's loss as \text{loss}
     instead of \mathcal{L}.

5. **misleading** - Qwen2-VL's packing is of images in the vision encoder, at inference; Molmo never calls its trick
   packing
   - Page: ch-training-step.html l.73 (5b panel): "the packing that Molmo and Qwen2-VL describe is in 5c and the text
     below." l.126: "Qwen2-VL, ... says that images of varying resolutions are packed into a single sequence; the
     paper states this for inference."
   - Source: Qwen2-VL template.tex l.357-360 (Sec. 2.1): "To support this feature, we modified ViT by removing the
     original absolute position embeddings and introducing 2D-RoPE ... At the inference stage, images of varying
     resolutions are packed into a single sequence, with the packed length controlled to limit GPU memory usage.
     Furthermore, to reduce the visual tokens of each image, a simple MLP layer is employed after the ViT". The
     sentence sits between the ViT change and the "after the ViT" merger, in a paragraph that cites NaViT (l.357),
     so it most plausibly means packing image patches into the vision encoder's input. It is not about batching
     training examples for the LLM, which is what this section is about. Molmo never uses the word "pack" (no match
     in mmolmo.tex); it says "we arrange all of text annotation tokens for an image in one long sequence" (l.194).
   - Replace with:
     l.73: "This is common practice rather than something these papers specify; Molmo's related trick for one image
     with many questions is in 5c, and Qwen2-VL's packing of images is in the text below."
     l.126: "...; the paper states this for inference, in its description of the vision encoder (NaViT-style packing
     of image patches), not for batching the language model's training examples."

6. **misleading** - the pipeline joins LLaVA-1.5's encoder to LLaVA v1's schedules
   - Page: viz-pipeline.js l.222/236/242 (stage 2): "2. Image to tokens (LLaVA-1.5 sizes)", "CLIP ViT-L/14, 336
     px", connector "2-layer MLP". ch-training-step.html l.147: "only the connector W learns ... for 1 epoch with a
     learning rate of 2e-3 and a batch size of 128"; l.148: "3 epochs, with a learning rate of 2e-5 and a batch size
     of 32"; viz-pipeline.js l.470: "trainable: W only". The panels are labelled per paper, but the
     "one step" story implies one model, so a reader carries the 2-layer MLP into stage 7.
   - Source: LLaVA v1's connector is "a simple linear layer ... a trainable projection matrix W" (LLaVA
     approach.tex l.104-106), at 224 px (LLaVA-1.5 3_scaling.tex l.228, row 0: "LLaVA & 7B & 224"); the
     2e-3/128 and 3-epoch 2e-5/32 schedules are v1's (experiment.tex l.46). LLaVA-1.5 swaps in "a two-layer MLP"
     (3_scaling.tex l.255) and CLIP-ViT-L-336px (l.262), and its own table gives pretrain lr 1e-3, batch 256;
     finetune lr 2e-5, batch 128; 1 epoch each (9_appendix.tex l.155-160). (Its text, l.145, says "the same set of
     hyperparameters as the original LLaVA, except that we halve the learning rate in pretraining", which does not
     match its own table's batch sizes and epochs; not worth chasing on the page.)
   - Replace with (append to l.148, and needs a new claim for the LLaVA-1.5 table): "LLaVA-1.5 keeps the same two
     stages, with 558K pairs for the first and its 665K mixture for the second, but its connector is the 2-layer MLP
     drawn in stage 2 rather than the original single matrix W, and its own schedule is one epoch per stage
     (learning rate 1e-3 and batch 256, then 2e-5 and batch 128)."

7. **unlabelled** - "at real scale" on toy-derived percentages
   - Page: ch-training-step.html l.130: "at real scale, with 576 visual tokens per image, it is 2.5%. Most of what
     the model reads, it is never graded on." viz-pipeline.js l.493: "At real scale (576 visual tokens) only 2.5% of
     the padded batch carries loss." l.120: "At real scale the picture dominates that sequence: our three short
     examples become 603, 617 and 599 tokens long".
   - Source/check: 2.5% = 47 / (3 x 617) keeps all the toy text. The whole system message is one "SYS" token
     (viz-pipeline.js l.12), although LLaVA uses Vicuna-v0's system message (Table 2 caption, approach.tex l.159).
     The text is one word per token, and the answers are toy-short (C's is one word). Only the 576 is real. The
     caption's toy list (l.96) says "every length and percentage computed from them", but "at real scale" reads as
     a statistic about real data. The 603/617/599 and 1.7% sentences are saved by "our three short examples" and
     "the same three examples".
   - Replace with:
     l.130: "...; keep the same toy text but give each image its real 576 visual tokens and it is 2.5%. The exact
     figure depends on the toy's short answers, but the direction holds: most of what the model reads, it is never
     graded on."
     JS l.493: "With real-size images (576 visual tokens) but the same toy text, only 2.5% of the padded batch
     carries loss."
     l.120: "With real-size images the picture dominates that sequence: ..."
     Caption l.96, toy list: add "the whole system message as one token".

8. **unlabelled** - the <image> placeholder and the splice step are not in the LLaVA papers
   - Page: ch-training-step.html l.61 (3 Text): "The conversation is poured into LLaVA's template (its Table 2): ...
     An <image> placeholder marks where the picture goes." l.65 (4 Splice): "The placeholder is cut out and the
     visual tokens are dropped into its place." l.119: "with an <image> placeholder in the first human turn". The
     caption (l.96) lists this neither as sourced nor as common practice.
   - Source: Table 2 writes X_instruct^1 (approach.tex l.147), and Eq. 2 puts X_v itself into it (l.127-134).
     Neither LLaVA nor LLaVA-1.5 mentions an <image> token or a placeholder (grep over both: no match). Flamingo
     does insert "<image> tags in plain text" (content.tex l.227), but it conditions through cross-attention and
     does not splice.
   - Replace with (caption l.96): "Common practice, not specified by these papers: an <image> placeholder token that
     is replaced by the visual tokens (LLaVA's paper just writes the image X_v into the first instruction), padding
     with a pad mask, and packing unrelated examples with a block-diagonal mask." Optionally, at l.61: "An <image>
     placeholder (an implementation device; the paper writes the image X_v straight into the first instruction)
     marks where the picture goes."

9. **unlabelled** - the reason given for accumulate beating round-robin is the page's own
   - Page: ch-training-step.html l.116: "One way to see why: when every update sees all the datasets at once, no
     single dataset gets to pull the weights its own way for a whole update."
   - Source: Flamingo gives no reason: "We accumulate gradients over all datasets, which we found outperforms a
     round-robin approach" (content.tex l.248); "We show in row (ii) the importance of our gradient accumulation
     strategy compared to using round-robin updates" (l.346).
   - Replace with: "The paper does not say why. One plausible reading (ours): when every update sees all the
     datasets at once, no single dataset gets to pull the weights its own way for a whole update."

10. **nit** - Molmo equivalence made "exact"; "for our data mix" dropped
    - Page: ch-training-step.html l.126: "That is exactly equivalent to separate examples, and it ends up reducing ...
      at the price of only a 25% increase in sequence length".
    - Source: mmolmo.tex l.194: "This setup is equivalent to training on individual image-text pairs but avoids
      redundant image encoding, reducing the number of processed images by two-thirds and shortening training time
      by over half, with only a 25% increase in sequence length for our data mix." The paper says nothing about
      position ids. With RoPE, a later annotation sits further from the image than it would on its own, so "exactly"
      goes beyond the paper.
    - Replace with: "The paper calls this equivalent to training on individual image-text pairs, and it ends up
      reducing the number of processed images by two-thirds and shortening training time by over half, at the price
      of only a 25% increase in sequence length for Molmo's data mix (Molmo caps sequences at 2304 tokens)."
      (Keep the existing claim spans.)

11. **nit** - Molmo normalisation result loses its hedge
    - Page: ch-training-step.html l.144: "and reports that captioning performance drops by 0.5-1 points without this
      fix."
    - Source: mmolmo.tex l.864: "We observe that captioning performance can drop by 0.5-1 points without this fix."
    - Replace with: "and reports that captioning performance can drop by 0.5-1 points without this fix."

12. **nit** - Molmo's square-root mixture omits the manual re-weighting
    - Page: ch-training-step.html l.104: "Molmo samples each dataset at a rate proportional to the square root of
      their size, which gives small datasets a larger share than their size alone would."
    - Source: mmolmo.tex l.249: "We sample datasets at rates proportional to the square root of their size, with
      manual down weighting of some very large synthetic datasets (PlotQA, FigureQA, DVQA, and PixMo-Clocks). We
      observe that pointing tasks learn more slowly than QA tasks, so we significantly up-weight the pointing data."
    - Replace with: "Molmo samples each dataset at a rate proportional to the square root of their size, then adjusts
      by hand (down-weighting some very large synthetic sets, up-weighting pointing data); the square root alone
      gives small datasets a larger share than their size would."

13. **nit** - LLaVA vs LLaVA-1.5 sampling: the contrast is the page's
    - Page: ch-training-step.html l.104: "LLaVA's three kinds of instruction data are uniformly sampled in training.
      LLaVA-1.5 concatenates its ten splits so that every example is sampled with the same probability, which means
      the big splits dominate." Also l.52: "each example is sampled with the same probability".
    - Source: LLaVA approach.tex l.188: "Among the three types of responses, ... They are uniformly sampled in
      training." (It does not say per type or per example.) LLaVA-1.5 9_appendix.tex l.141: "All data splits are
      concatenated together and sampled with the same probability." (The grammatical subject is the splits; "every
      example" is the natural reading after "concatenated", but it is a reading.) The side-by-side wording implies
      LLaVA balanced its three kinds and LLaVA-1.5 did not; neither paper establishes that.
    - Replace with (both claim spans keep their verbatim text): "LLaVA says only that its three kinds of
      instruction data are uniformly sampled in training, not whether per kind or per example. LLaVA-1.5
      concatenates its ten splits, which are then sampled with the same probability; after concatenation that makes
      every example equally likely, so the big splits dominate."

14. **nit** - "examples never interact" vs the page's own normalisation paragraph
    - Page: ch-training-step.html l.101: "Examples in a batch never interact: padded rows are independent, and packed
      rows are masked apart. The batch size only sets how noisy each gradient is and how busy the GPUs are."
    - Source: l.144 on the same page, and Molmo mmolmo.tex l.861, show the coupling. When the loss is divided by a
      count of loss tokens, each example's weight depends on the other examples' token counts ("effectively
      up-weights examples with a small number of loss tokens").
    - Replace with: "Examples in a batch never see each other: padded rows are independent, and packed rows are
      masked apart. Apart from how the loss is averaged (see the Molmo note below), the batch size mainly sets how
      noisy each gradient is and how busy the GPUs are."

15. **nit** - the round-robin definition comes from a different experiment; batch sizes also set Flamingo's mix
    - Page: ch-training-step.html l.116: "The obvious alternative, round-robin, is to alternate batches of each
      dataset, updating the parameters on each batch. It sounds equivalent, but it scores 70.7 vs 62.9 in Flamingo's
      ablation". l.105: "Flamingo does not mix inside a batch at all. It writes the mixture into the loss:"
    - Source: the quoted definition is from App. B.3.2 "Dataset mixing strategies for the contrastive pretraining"
      (B_experiments.tex l.368-381, a small NFNet-F0 + BERT-mini contrastive model, l.375). The VLM ablation behind
      70.7 vs 62.9 only cites Cho et al. for "round-robin" (content.tex l.248, l.346). The VLM ablation also fixes
      per-dataset batch sizes, "256 for M3W, 512 for ALIGN, 512 for LTIP and 64 for VTP" (B_experiments.tex l.310).
      Those batch sizes set how much of each dataset an update sees, and lambda_m sets how much each dataset's loss
      counts.
    - Replace with: l.116: "The obvious alternative, round-robin, is (in the words of Flamingo's appendix, which
      compares the same strategies for its contrastive encoder) to alternate batches of each dataset, ..." l.105:
      "Flamingo does not mix inside a batch at all: each update takes a fixed-size batch from every dataset, and the
      mixture is written into the loss:"

16. **nit** - "158K instruction conversations"
    - Page: ch-training-step.html l.93 (7b panel): "on the 158K instruction conversations"; viz-pipeline.js l.470:
      "data: LLaVA-Instruct-158K conversations".
    - Source: approach.tex l.78: "We collect 158K unique language-image instruction-following samples in total,
      including 58K in conversations, 23K in detailed description, and 77k in complex reasoning". "Conversation" is
      LLaVA's name for one of the three kinds.
    - Replace with: "on the 158K instruction-following samples (conversations, detailed descriptions and complex
      reasoning)"; JS: "data: LLaVA-Instruct-158K (conversation, detail, reasoning)".

17. **nit** - stage-1 readout forgets the <STOP>s
    - Page: viz-pipeline.js l.494: "Stage 1: the loss is computed on the caption tokens; ..."
    - Source: stage-1 samples use the Table 2 format (approach.tex l.177), where every <STOP> is green (l.146-150).
      Sec. 3 gives the stage-1 expansion as "Human: X_q X_v <STOP> Assistant: X_c <STOP>" (approach.tex l.9-13).
    - Replace with: "Stage 1: the loss is computed on the caption tokens and the <STOP>s; ..."

18. **nit** - hedge dropped on "visual tokenizer"
    - Page: ch-training-step.html l.89: "which the paper describes as training a compatible visual tokenizer for the
      frozen LLM."
    - Source: approach.tex l.177: "This stage can be understood as training a compatible visual tokenizer for the
      frozen LLM."
    - Replace with: "which, the paper says, can be understood as training a compatible visual tokenizer for the frozen
      LLM."

19. **nit** - "the language model has no idea which vectors came from pixels"
    - Page: ch-training-step.html l.65 (4 Splice): "From here on, the language model has no idea which vectors came
      from pixels and which from words."
    - Source: LLaVA only makes the widths equal (approach.tex l.106), and nothing in its sequence marks the image
      block. Later models mark it explicitly: Qwen2-VL "employs special tokens to distinguish vision and text inputs"
      (template.tex l.398) and gives visual tokens their own M-RoPE ids (l.369).
    - Replace with: "From here on, nothing in LLaVA's sequence marks which vectors came from pixels and which from
      words (later models add markers, such as Qwen2-VL's vision-start and vision-end tokens)."

20. **nit** - "Skipping stage 1 is not free" generalises one ScienceQA run
    - Page: ch-training-step.html l.148: "Skipping stage 1 is not free: on ScienceQA, training without it reaches
      85.81% against 90.92% with it."
    - Source: experiment.tex l.233: "We skip pre-training and directly train on Science QA from scratch --
      performance drops to 85.81% accuracy." The ScienceQA runs fine-tune on ScienceQA for 12 epochs (l.162), not
      on the 3-epoch 158K schedule given in the previous sentence. Molmo reports the opposite in its setting: "We
      find this step unnecessary when pre-training on PixMo-Cap" (mmolmo.tex l.230-231).
    - Replace with (needs claims for "12 epochs" and the Molmo quote): "Skipping stage 1 was not free for LLaVA: on
      ScienceQA (fine-tuned on ScienceQA itself for 12 epochs), training without it reaches 85.81% against 90.92%
      with it. The lesson is not universal: Molmo finds a connector-only stage unnecessary when it pre-trains all
      weights on its dense captions."

21. **nit** - closing sentence contradicts the chapter's own Qwen2-VL and Molmo mentions
    - Page: ch-training-step.html l.150: "Notice, though, that every image in this chapter was a small square, 336
      pixels on a side at most."
    - Source: the same chapter cites Qwen2-VL (l.126), which "can now process images of any resolution"
      (template.tex l.357), and Molmo, whose images are multi-crop.
    - Replace with: "Notice, though, that every image in this chapter's pipeline was a small square, 336 pixels on a
      side at most."

22. **nit** - template panel: "nothing yet marks which is which"
    - Page: viz-pipeline.js l.271: "Human text and the assistant's answers are both plain tokens: nothing yet marks
      which is which."
    - Source: the role labels (Table 2, approach.tex l.147-148) do mark who speaks. What nothing marks yet is which
      tokens will be graded.
    - Replace with: "Human text and the assistant's answers are both plain tokens: nothing yet marks which ones will
      be graded."

23. **nit** - toy example C goes against LLaVA-1.5's QA merging
    - Page: viz-pipeline.js l.18 (C: one VQAv2 question, one-word answer), l.196 "from VQAv2"; ch-training-step.html
      l.52 "one from VQAv2"; l.144 "like example C's single word".
    - Source: LLaVA-1.5 9_appendix.tex l.132: "For all VQA datasets, QA pairs from the same training image are
      merged into a single conversation." So a VQAv2 sample in this mix is multi-turn.
    - Replace with (caption l.96, toy list): "..., and example C as a single VQAv2 question (LLaVA-1.5 merges all QA
      pairs of a VQA image into one conversation)".

24. **nit** - 6b's two views mark red one position apart
    - Page: viz-pipeline.js l.423-426 (DRAW.mask). The bottom strip colours a cell red if the token in it is a loss
      token, which puts red on the target token's own cell. The top row (drawTargets l.395) colours the target chip
      under the position that predicts it. The count is the same (47), but the red cells are shifted one place
      between the two views.
    - Replace with: label "whole padded batch: 47 of 150 cells are graded target tokens (31%)", or colour cell i by
      isLoss(row[i + 1]) so both views mark the predicting positions.

25. **nit** (claims file, not page text) - llava-cc3m-filtered loc points at the wrong appendix section
    - Claim: notes/claims/llava.json (merged in notes/claims.json), id llava-cc3m-filtered, loc "Sec. 4.2 Stage 1;
      Appendix Assets CC3M". Used on the page at ch-training-step.html l.89 and l.147.
    - Source: the Sec. 4.2 half is right (approach.tex l.176). The appendix sentence "This results in around 595K
      image-text pairs." is under \section{Data}, \paragraph{CC3M.} (appendix.tex l.184, l.262-264). The Assets
      section (l.170-183) lists repository links and does not mention CC3M or 595K.
    - Replace with: loc "Sec. 4.2 Stage 1; Appendix Data, CC3M paragraph".

## Checked and found correct

- All 34 data-claim values match the sources (LLaVA, LLaVA-1.5, Flamingo, Molmo, Qwen2-VL, CLIP), and all locs do
  except the appendix half of llava-cc3m-filtered (item 25).
- LLaVA Table 2 (rendered PDF p.5; approach.tex l.146-159): green = X_a^t and every <STOP>, including the ones after
  the system message and after each Human turn. "Human:", "Assistant:", the system message and X_instruct are not
  green. <STOP> = ###. "trained to predict the assistant answers and where to stop". It is Table 2 in the PDF.
- LLaVA Eq. 3 (approach.tex l.168-170) and Flamingo Eq. 2 (content.tex l.242-244) match symbol by symbol. Equation
  numbers (LLaVA (1)(2)(3), Flamingo (1)(2)) were confirmed in the PDFs. LLaVA's Eq. 2 randomises [X_q^1, X_v] vs
  [X_v, X_q^1] in the first turn only, and toy A has the image after the question.
- Flamingo mixture: the lambda_m are loss weights, not sampling rates ("minimizing a weighted sum of per-dataset
  expected negative log-likelihoods", content.tex l.241). The accumulation bullet says "These gradients are then
  weighted and summed" (B_experiments.tex l.381). Gradients are accumulated, one batch per dataset, then one update
  (content.tex l.248; B_experiments.tex l.311: "1 million gradient steps (meaning 250,000 gradient updates ... as we
  accumulate gradients over four datasets)"). Weights are 1.0/0.2/0.2/0.03 for M3W/ALIGN/LTIP/VTP (l.79), M = 4.
  "key to performance" is at content.tex l.247. 70.7 vs 62.9 is the Flamingo-3B baseline vs Table 3 row (ii)
  Round Robin, Overall score (ablations_no_classif.tex l.8, l.13).
- LLaVA freezing: stage 1 freezes the encoder and LLM and trains only W (approach.tex l.177). Stage 2 trains
  {W, phi}, and the encoder is always frozen (l.180). Schedules are 1 epoch/2e-3/128 and 3 epochs/2e-5/32
  (experiment.tex l.46; appendix l.166), a 100x LR ratio. 595K is at approach.tex l.176, 158K at l.78. 85.81 vs
  90.92 is at experiment.tex l.162, l.233.
- "Gradient passes through the frozen LLM to reach the connector" holds. H_v enters at the LLM's input embeddings
  (Eq. 1), so the gradient for W must backpropagate through every LLM layer's activations; frozen layers need no
  weight gradients. Nothing below W trains, so the encoder needs no backward pass, which agrees with Flamingo
  B_experiments.tex l.66.
- Molmo: the mask rule and the savings quote are verbatim (mmolmo.tex l.194), and 2304 is at l.866. The paper does
  not specify attention inside the image block, and the page labels that as an assumption. The loss-normalisation
  direction is right: few loss tokens on a device means a small local divisor, which up-weights those examples. The
  fix divides by the average count across devices (l.861-862), and the page's rationale matches.
- LLaVA-1.5: the ten split sizes (158/86/83/80/72/66/48/40/22/9K; 9_appendix.tex l.76-93) match the figure bars and
  the 665K total. Single-modality batches give +25% (l.138), QA merging is at l.132, and 558K/665K at 3_scaling.tex
  l.296. 336 px from CLIP-ViT-L-336px (l.262) gives 336/14 = 24 and 24 x 24 = 576.
- Qwen2-VL: 224 px -> 66 tokens (template.tex l.361: 16 x 16 = 256 patches, 2 x 2 merge -> 64, + vision start/end
  = 66). The inference qualifier is on the page.
- CLIP: minibatch 32,768 (clip_paper.tex l.263, Sec. 2.5).
- Toy recompute from viz-pipeline.js definitions (node):
  - Lengths: raw 28/42/24 -> spliced 36/50/32, LMAX 50, packed total 118.
  - Padding: 32 of 150 cells = 21.3%.
  - Loss tokens: 11 + 32 + 4 = 47 of 150 = 31.3%.
  - Real-size images: 603/617/599 tokens; padding 32/1851 = 1.73% (shown as 1.7%); loss 47/1851 = 2.54% (shown as
    2.5%).
  - Molmo toy: QA lengths 14/12/10 -> 63 tokens with 3 encodes vs 45 with 1.
  - All match what the page and readouts display.
- Labels present and correct: packing marked as common practice (5b panel, canvas heading, caption, l.124); padding
  marked common practice in the caption; the Molmo image-block causal assumption is marked; the Qwen2-VL statement
  is marked as inference; the caption notes green in the paper vs red on the page.

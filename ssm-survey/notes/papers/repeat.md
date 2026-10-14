# Repeat After Me (recall group, serves ch 10)

## 1. Identity
- Title: "Repeat After Me: Transformers are Better than State Space Models at Copying".
- arXiv 2402.01032, version read: v2 (2024-06-03, abs.html `citation_online_date`). v1 submitted 2024-02-01.
- Authors: Samy Jelassi (first) ... Eran Malach (last); with David Brandfonbrener, Sham M. Kakade. Harvard
  (CMSA, Kempner Institute, SEAS).
- Venue: ICML 2024 (`\usepackage[accepted]{icml2024}` in `src/main.tex`).
- Code: https://github.com/sjelassi/transformers_ssm_copy (Sec 3.1).
- Sources read: `src/main.tex`, `src/icml2024/{introduction, theoretical_motivation, experiments,
  appendix_exp, appendix_proof (statements and lower-bound proof), discussion}.tex`; `paper.pdf` pp. 2-8
  for figure and theorem numbers.

## 2. Core problem
Models with a fixed-size state (SSMs, RNNs, linear attention; the paper calls them GSSMs) are cheap at
inference. What do they give up? The paper tests copying and retrieval from the context.

## 3. What prior methods did and why that hurt (per the paper)
- GSSMs (S4, Mamba, LSTM, linear attention, RWKV, RetNet) keep O(1) memory per token. Their language-model
  perplexity looked competitive, so the cost was unclear (Sec 1).
- Perplexity hides the gap: Mamba has lower Pile perplexity than Pythia at matched size yet copies and
  retrieves worse (Sec 1, Sec 4.1).

## 4. The one key idea
Copying a string of L tokens needs about L log D bits of memory. A GSSM's state is fixed, so it must fail
once L log D exceeds the state; a 2-layer transformer can look back with n-gram hashing and copy strings of
length exponential in its size.

## 5. Mechanism, step by step
1. Define GSSM: finite state set S, update u: S x D -> S, output r: S -> D; mem(S) = log|S| bits (Sec 2.1).
   Parameters are unconstrained; only the input-dependent memory is bounded (Remark 2.1).
2. Define the copy task: `<BOS> x_1 ... x_L <COPY>`, then output x exactly (Sec 2.1, Fig 2).
3. Upper bound (Thm 2.3): a depth-2 transformer with Hard-ALiBi heads hashes the last n tokens, attends to
   the earlier occurrence of that n-gram, and emits the next token (Fig 3, Algorithm in App). Fails only
   when an n-gram repeats (Def 2.2, Lemma 2.4).
4. Lower bound (Thm 2.7, Cor 2.8): counting argument; the output after `<COPY>` is a function of the state
   alone (lemma `lem:fixed_state`, appendix "Proofs - Lower Bound"), so at most |S| distinct strings
   can be produced.
5. Experiments from scratch (~160M params): data efficiency, length generalization, n-gram probe,
   n-gram lookup (suffix and prefix keys).
6. Pretrained models: Pythia (410M-2.8B) vs Mamba (360M-2.8B), both trained on the Pile with the same
   tokenizer: copy C4 text, copy shuffled text, phone-book lookup, SQuAD QA.

## 6. Equations / statements (Sec 2; verbatim TeX from `theoretical_motivation.tex`)
Macros: `\sD` dictionary (D alphabet tokens), `\gS` state space, `\mem` memory in bits, `\gD_L` copy
distribution, `\err` error, `\ngram` n-gram.

**Copy error (Sec 2.1).**
```
\err_{\gD_L}(H) = \Pr_{\gD_L}\left[H_{1:L}(\BOS, \vx, \COPY) \ne \vx\right]
```
Gloss: probability that the first L generated tokens are not an exact copy of x.

**Hard-ALiBi (Sec 2.2).** `\vo_i = V_i \cdot \softmax(K_i \cdot \vq_i + b_i)` with `b_{i,j} = -\infty` for
`j \le i-m` and `0` for `j > i-m`. Gloss: head sees only its last m tokens; m = infinity gives a plain NoPE
head.

**Definition 2.2.**
```
p_{\ngram}(\gD_L) = \Pr_{\gD_L} \left[\exists_{i \ne j}~\mathrm{s.t.}~x_i, \dots x_{i+n} = x_{j}, \dots, x_{j+n}\right]
```
Gloss: chance that some n-gram appears twice in the string.

**Theorem 2.3 (p.4).**
```
For all $n$, there exists a depth-2 transformer $\gT$ of dimension $O(n\log(D))$ s.t. for all $2n \le L \le D^n$, and for any copy distribution $\gD_L$, $\err_{\gD_L}(\gT)< p_{\ngram}(\gD_L)$.
```
Plain words: 2 layers and width O(n log D) copy any string up to length D^n, failing only on repeated
n-grams. The first block uses n Hard-ALiBi heads (App, proof of Thm 2.3, "a combination of n hard-ALiBi
attention heads"), and the second block one NoPE head. So copyable length D^n is exponential in the number
of heads n; the abstract and Sec 1 say "exponential in the number of heads".

**Lemma 2.4.** `p_{\ngram}(\gD_L) < L^2 D^{-n}` for uniform strings.

**Corollary 2.5.** Error < epsilon with a depth-2 transformer of dimension `O(\log(L/\epsilon)\log(D))`:
parameters logarithmic in L.

**Theorem 2.7 (p.4).**
```
Fix some GSSM $H$ over state space $\gS$. Then, for all $L$, for the uniform copy distribution $\gD_L$, the model $H$ has error $\err_{\gD_L}(H) > 1-\frac{\abs{\gS}}{D^L}$.
```
Plain words: a state that can take |S| values can reproduce at most |S| of the D^L possible strings.

**Corollary 2.8 (p.4).**
```
Fix some $L \in \sN$. Then, every GSSM $H$ with state space $\gS$ s.t. $\mem(\gS) < L\log(D)-1$ has error $\err_{\gD_L}(H) > 1/2$ for the uniform copy distribution $\gD_L$.
```
Plain words: fewer than L log2 D - 1 bits of state means more than half of random strings are copied wrong.
This is the closest formula in the recall group to "how much a state of a given size can store".

**Remark 2.9.** Transformers copy length-L inputs with O~(L) input-dependent memory, which Cor 2.8 shows is
optimal up to log factors.

Worked capacity example (Sec 3, first paragraph): "a latent state of 1000 32-bit floating point numbers has
enough bits to store at least 2000 tokens from a 50K token vocabulary." (Our check: 32,000 bits /
log2(50,000) ~ 15.6 bits per token gives ~2,050 tokens.)

## 7. Measurement setup (analysis lens)
From-scratch synthetic (Sec 3.1, App A.2):
- Transformers: GPT-NeoX, 12 layers, hidden 1024, H = 16 heads; positional variants RoPE, NoPE, ALiBi,
  Hard-ALiBi (best M = 6 masked heads from sweep M in {2..10}).
- Mamba: 24 layers, hidden 1024; state dim swept over {16, 32, 64, 128, 256}, best S = 32. Both ~160M params.
- LSTM: largest trainable ~40M params.
- Vocabulary 30 tokens (a-z + BOS, EOS, COPY); uniform strings; context packed with i.i.d. examples;
  context length C = 420 (C = 620 for Fig 1a).
- Training: online batch 64 per epoch; 15 epochs (transformers, Mamba), 300 (LSTM); AdamW, LR 5e-5 (sweep
  {5e-5, 1e-4, 5e-4} for LSTM and Mamba), linear decay, 300 warmup steps, weight decay 0.1. RTX8000 GPUs,
  ~600 GPU hours. Eval: 10 batches of 128; string-level accuracy; greedy decoding.
Pretrained (Sec 4.1):
- Pythia 410M, 1.4B, 2.8B vs Mamba 360M, 1.4B, 2.8B; all Pile-trained, same tokenizer. Training token
  counts are not stated in this paper.
- Held fixed: data (Pile) and tokenizer; only architecture differs. Eval over 10 batches of 64 (SQuAD:
  50 questions).
- Phone book: L random (name, number) lines like "John Powell: 609-323-7777", two few-shot examples, one
  query; L varies 20-200 (Fig 1c x-axis).
- Falsifier: a Mamba matching Pythia on phone-book accuracy at L >= 70, or a GSSM copying uniform strings
  with mem(S) < L log D.

## 8. Results worth remembering
- Fig 1a (p.2): train on strings <= 300, test at 300; "transformers need 100x less samples than the best
  GSSMs" (Sec 3.2). LSTM never learns it. Fig 9b (App B.1): at length <= 30, Mamba needs 10x more data, LSTM
  100x more.
- Fig 1b (p.2): train on <= 50 tokens, test up to 1000; GSSMs (LSTM, Mamba) "drop to zero almost
  immediately"; Hard-ALiBi transformer "almost perfect length generalization up to sequences of length 1000"
  (Sec 3.3).
- Fig 4 (p.5): Hard-ALiBi transformer accuracy holds for duplicated n-grams n <= 4, drops from n >= 5:
  "relies on something like 5-gram retrieval" (Sec 3.4). Fig 11: matches a perfect 5-gram model.
- Fig 5 (p.6): suffix-key n-gram lookup, trained <= 30 tokens: transformers hold up to length 100; GSSMs
  "perform poorly beyond their training distribution".
- Fig 6 (p.6): prefix-key variant (key given first): GSSMs reach "perfect length-generalization" and beat
  NoPE and ALiBi transformers (not Hard-ALiBi). This is the paper's own counter-case: a summary-sized task.
- Fig 1c (p.2) phone book: "even the smallest transformer (410M parameters) outperforms the largest GSSMs
  (2.8B parameters) when the phone-book size is long enough (L >= 70)" (Sec 4.3). Visual read: Pythia 1.4B
  and 2.8B stay near 90-100% up to L = 200; Pythia 410M falls to about 50%; Mamba models fall to about 30%
  or lower by L = 100 and near 0 by L = 200.
- Fig 7a (p.8): copy C4 strings; "even the smallest transformer model dramatically outperforms the largest
  GSSM" (Sec 4.2).
- Fig 7b (p.8): shuffled words; "Even the largest GSSM now gets zero accuracy on strings of length 300".
- Fig 7c (p.8): SQuAD F1 vs paragraph length, 2.8B models: similar on short paragraphs; Mamba decays faster.
- Fig 10 (App B.2): uniform random strings; smallest Pythia beats largest Mamba; Pythia 1.4B "almost 100%".

## 9. Limitations the paper states
- Theory is about representation, not learning (Sec 1); a large GSSM may have enough bits in principle
  (Sec 3).
- GSSMs win when a small summary suffices (prefix-key lookup, Fig 6; flip-flop language modeling cited in
  Related Work).
- Discussion: future work should build hybrids that give SSMs attention-like retrieval. Later evidence in
  our sources: NVIDIA 8B study (2406.07887) shows a Mamba-2-Hybrid with 4 attention layers solves Phonebook
  to and beyond its context length; Jamba likewise.

## 10. Outline feedback
- Ch 10 intuition line "a phone number seen 10K tokens ago" is not from this paper. Here the phone book goes
  to L = 200 entries and the gap opens at L >= 70. NVIDIA reports pure-SSM failure from ~500 tokens. Reword
  to sourced lengths.
- The paper's own definition of GSSM covers linear attention, RWKV, RetNet, LSTM, S4, Mamba. Use it for the
  ch 10 framing: "fixed-size state" is the class, not "SSM" narrowly.
- Mamba's state-dim sweep found S = 32 best of {16..256}: more state did not fix copying in training. Worth
  one sentence on the card (it complicates "just make N bigger").
- Cor 2.8 (L log2 D bits) is the formula to show beside `viz-phonebook`.

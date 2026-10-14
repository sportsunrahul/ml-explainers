# The Illusion of State in State-Space Models (recall group branch, serves ch 10 one-liner)

## 1. Identity
- Title: "The Illusion of State in State-Space Models".
- arXiv 2404.08819, version read: v3 (2025-03-05, abs.html). v1 2024-04-12; v2 2024-06-04.
- Authors: William Merrill (first, NYU), Jackson Petty (NYU), Ashish Sabharwal (last, AI2).
- Venue: ICML 2024 (`\usepackage[accepted]{icml2024}`, `src/SSM-NoStateTracking.tex` l.23).
- Sources read: `src/SSM-NoStateTracking.tex` (intro, Sec 2.1 definition, Sec 3, Sec 4 statements and
  discussion, Sec 6, conclusion); `paper.pdf` pp. 5-8 for numbering and Fig 3.

## 2. Core problem
SSMs are often motivated as RNN-like and so better at sequential state tracking than transformers. Is that
true for expressive power at fixed depth?

## 3. What prior methods did and why that hurt (per the paper)
- Transformers are known to lie in TC0 and cannot express inherently sequential problems (Merrill and
  Sabharwal 2023).
- LSSL (Gu et al. 2021, Lemma 3.2) claimed SSMs can simulate RNNs, but that needs infinite depth (Sec 1,
  Sec 4.4).

## 4. The one key idea
Linear SSMs (S4) and diagonal input-dependent SSMs (S6 / Mamba) can be simulated by L-uniform TC0
circuits, exactly like transformers. So with fixed depth they cannot solve NC1-complete state tracking such
as composing permutations of 5 objects (the S5 word problem), assuming TC0 != NC1.

## 5. Mechanism, step by step
1. Define a generalized linear SSM layer (Def 2.1) with recurrent and convolutional forms.
2. State tracking = word problem on a finite monoid (Def 3.1): multiply a sequence of elements.
3. Barrington: word problems of non-solvable groups (S5, A5) are NC1-complete.
4. Theorem 4.2 (non-gated SSM, e.g. S4) and Theorem 4.4 (diagonal SSM, e.g. S6): convolutional form is in
   L-uniform TC0 (log precision). Corollaries 4.3 (S4), 4.5 (S6 / Mamba).
5. Corollary 4.7: no log-precision S4 or S6 SSM solves the S5 word problem (if TC0 != NC1).
6. Extensions that escape TC0: a nonlinear RNN-SSM (Thm 5.1) or an input-dependent, non-diagonal transition
   (IDS4, WFA-SSM, Thm 5.2); Corollary 5.3: one layer suffices for S5.
7. Experiments on A5, A4 x Z5, Z60 word problems as token tagging.

## 6. Equations
**Generalized linear SSM layer (Def 2.1, Sec 2.1), verbatim:**
```
\mathbf h_i = \barA_i \mathbf h_{i-1} + \barB_i \mathbf x_i .
\tag{Recurrent form}
```
```
\mathbf h_i = \sum_{j=1}^i \left( \prod_{k=j+1}^i \barA_k \right) \barB_j \mathbf x_j .
\tag{Convolutional form}
```
Gloss: notation already matches the page (x input, h state of size d here, A-bar, B-bar per step; output
y_i = C_i h_i + D_i x_i). The products of A-bar run from greatest to least index.

**Corollary 4.7 (p.7), verbatim:** "Assuming $\TC^0 \neq \NC^1$, no log-precision SSM with the S4 or S6
architecture can solve the word problem for $S_5$ or any other $\NC^1$-hard problem."

## 7. Measurement setup (analysis lens)
- Task (Sec 6): token tagging; input g_0 g_1 ... g_n from A5 (non-solvable, NC1-complete), A4 x Z5
  (non-abelian solvable), Z60 (abelian); all three groups have 60 elements; label at step i = product of the
  first i elements. All 3600 length-2 sequences are in training.
- Models: transformer, RNN, S4, Mamba, IDS4.
- Metric: minimum depth reaching > 90% accuracy, plotted against sequence length (Fig 3, p.8).
- Falsifier: a fixed-depth Mamba that solves A5 at all lengths.

## 8. Results worth remembering
- Fig 3 (p.8): RNN and IDS4 need 1 layer at every length shown (up to 20) for all three groups; transformer,
  S4 and Mamba need depth "monotonically increasing in sequence length" on A5 and A4 x Z5.
- Our reading of Fig 3 (A5 panel): Mamba climbs from 1 to 4 layers by length ~12; the y-axis tops at 4.
- S4 and Mamba are "empirically better than transformer at approximate state tracking on the
  non-commutative tasks": the transformer needs at least as many layers (Sec 6, conclusion 3).

## 9. Limitations the paper states
- Worst-case expressivity at log precision and fixed depth; says nothing about typical inputs.
- IDS4 (input-dependent non-diagonal transitions) escapes TC0 but raises parallelism and learning-dynamics
  questions (Sec 5.3). Later in our sources: Gated DeltaNet and Mamba-3 move in this direction (check their
  notes; not claimed here).

## 10. Outline feedback
- This is a state-tracking claim, not a recall claim. On the ch 10 card, keep it separate from the
  phonebook/MQAR cards: one sentence ("Mamba and S4 cannot compose permutations at fixed depth, just like
  transformers") plus the Fig 3 result.
- Balance: the same figure shows Mamba beats the transformer at approximate state tracking. Mention it.

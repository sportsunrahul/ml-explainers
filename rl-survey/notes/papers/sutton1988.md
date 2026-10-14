# Sutton 1988: Learning to Predict by the Methods of Temporal Differences

Source: `sources/classic/sutton1988-td.pdf` (scanned journal article, 37 PDF pages; journal page = PDF page + 8,
so journal p.19 is PDF page 11). Page numbers below are journal pages. The OCR layer is noisy; everything below
was checked on the page images.

## 1. Identity
- Richard S. Sutton (sole author), GTE Laboratories Incorporated, Waltham, MA.
- "Learning to Predict by the Methods of Temporal Differences", Machine Learning 3: 9-44, 1988, Kluwer Academic
  Publishers, Boston (header, p.9). Received April 22, 1987; revised February 4, 1988.
- S&B's reference list (p.511, PDF 533) gives "Machine Learning, 3(1):9-44 (important erratum p. 377)" and spells
  the title "Learning to predict by the method of temporal differences" (singular "method"); the article itself
  prints "Methods".
- Sections: 1 Introduction p.9; 2 Temporal-difference and supervised-learning approaches to prediction p.11 (2.1
  Single-step and multi-step prediction p.12; 2.2 Computational issues p.12; 2.3 The TD(lambda) family p.15); 3
  Examples of faster learning with TD methods p.16 (3.1 game-playing example p.16; 3.2 random-walk example p.19);
  4 Theory p.23 (4.1 Convergence of linear TD(0) p.23; 4.2 Optimality and learning rate p.29; 4.3 TD methods as
  gradient descent); 5 Generalizations of TD(lambda); 6 Related research; 7 Conclusion p.40.

## 2. Core problem
Learning to predict in multi-step prediction problems, where "correctness is not revealed until more than one step
after the prediction is made, but partial information relevant to its correctness is revealed at each step" (p.12).

## 3. What prior methods did and why that hurt
Supervised learning pairs each observation with the final outcome z. The Widrow-Hoff update depends on z, "and thus
cannot be determined until the end of the sequence ... all observations and predictions made during a sequence must
be remembered until its end ... In other words, (2) cannot be computed incrementally" (p.14). Earlier TD uses
(Samuel's checker player, Holland's bucket brigade, Sutton's Adaptive Heuristic Critic) "have remained poorly
understood" (abstract), and "hitherto no TD method has ever been proved stable or convergent to the correct
predictions" (p.23).

## 4. The one key idea
Abstract, key sentence (p.9): "Whereas conventional prediction-learning methods assign credit by means of the
difference between predicted and actual outcomes, the new methods assign credit by means of the difference between
temporally successive predictions."

## 5. Mechanism
1. Data: observation-outcome sequences x_1, x_2, ..., x_m, z; predictions P_t = P(x_t, w) estimate z (p.13).
2. Weights change once per sequence by the sum of per-step increments (Eq. 1).
3. Supervised (Widrow-Hoff) increment uses z - P_t (Eq. 2).
4. Write z - P_t = sum_{k=t}^{m} (P_{k+1} - P_k), with P_{m+1} = z; regroup to get an incremental rule (Eq. 3) that
   gives the same total change (TD(1)).
5. Weight past gradients by lambda^{t-k} (Eq. 4) to get TD(lambda); lambda = 0 gives TD(0).

## 6. Equations (printed numbers; transcribed from page images)
- Eq. (1), p.13: `w \leftarrow w + \sum_{t=1}^{m} \Delta w_t`
- Eq. (2), p.13: `\Delta w_t = \alpha (z - P_t) \nabla_w P_t` (prototypical supervised-learning update; linear case
  is the Widrow-Hoff rule `\Delta w_t = \alpha (z - w^T x_t) x_t`, p.14).
- Eq. (3), p.15: `\Delta w_t = \alpha (P_{t+1} - P_t) \sum_{k=1}^{t} \nabla_w P_k` (TD(1)).
- Eq. (4), p.15: `\Delta w_t = \alpha (P_{t+1} - P_t) \sum_{k=1}^{t} \lambda^{t-k} \nabla_w P_k`, for
  0 <= lambda <= 1. Incremental trace (p.16): `e_{t+1} = \nabla_w P_{t+1} + \lambda e_t`.
- TD(0), unnumbered, p.16: `\Delta w_t = \alpha (P_{t+1} - P_t) \nabla_w P_t`. "The two equations are identical
  except that the actual outcome z in (2) is replaced by the next prediction P_{t+1}".
- Eq. (5), p.24: ideal predictions `E\{z \mid i\} = [\sum_{k=0}^{\infty} Q^k h]_i = [(I - Q)^{-1} h]_i`.
- Eq. (8), p.30: optimal (maximum-likelihood) predictions `[(I - \hat{Q})^{-1} \hat{h}]_i`.
- Notation vs S&B: no per-step reward in the base setting; a single outcome z ends each sequence; x_t is an
  observation vector, P_t a prediction, w the weights. S&B's tabular TD(0) is the case of unit-basis x_t.

Theorems (statements as printed):
- Theorem 1 (p.15): "On multi-step prediction problems, the linear TD(1) procedure produces the same per-sequence
  weight changes as the Widrow-Hoff procedure."
- Theorem 2 (p.24-25): "For any absorbing Markov chain, for any distribution of starting probabilities mu_i, for any
  outcome distributions with finite expected values z-bar_j, and for any linearly independent set of observation
  vectors {x_i | i in N}, there exists an epsilon > 0 such that, for all positive alpha < epsilon and for any
  initial weight vector, the predictions of linear TD(0) (with weight updates after each sequence) converge in
  expected value to the ideal predictions (5)."
- Theorem 3 (p.31): "For any training set whose observation vectors {x_i | i in N} are linearly independent, there
  exists an epsilon > 0 such that, for all positive alpha < epsilon and for any initial weight vector, the
  predictions of linear TD(0) converge, under repeated presentations of the training set with weight updates after
  each complete presentation, to the optimal predictions (8)." Optimal = maximum-likelihood estimate of the Markov
  process (p.29-30). This is the result S&B Section 6.3 calls certainty equivalence.

## 7. Measurement setup: the bounded random walk (Section 3.2, p.19-23)
- States A B C D E F G in a row; "Every walk begins in the center state D. At each step the walk moves to a
  neighboring state, either to the right or to the left with equal probability. If either edge state (A or G) is
  entered, the walk terminates." Outcome z = 0 at A, z = 1 at G.
- Nonterminal B..F use unit basis vectors of length 5 (tabular). True right-termination probabilities (p.20):
  1/6, 1/3, 1/2, 2/3, 5/6 for B, C, D, E, F.
- LABEL TRAP: S&B Example 6.2 calls the nonterminal states A..E and starts in C. Same walk, same true values,
  shifted letters. Do not mix the two labelings on the page.
- Data: "100 training sets, each consisting of 10 sequences, were constructed for use by all learning procedures"
  (p.20). lambda in {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1}; lambda = 1 is Widrow-Hoff.
- Experiment 1, repeated presentations (p.20-21): increments accumulated over a training set; each set presented
  until weights stopped changing. Measure: RMS error between asymptotic predictions and ideal predictions, averaged
  over training sets (Figure 3, p.20; caption: standard error about sigma = 0.01).
- Experiment 2, single presentation (p.22): each training set presented once; weights updated after each sequence;
  a range of alpha; all weights initially 0.5. Figure 4 (p.21): error vs alpha for several lambda, after 10
  sequences, averaged over 100 training sets. Figure 5 (p.22): error at the best alpha for each lambda.

## 8. Results and findings
- Experiment 1 (p.21): "performance improved rapidly as lambda was reduced below 1 (the supervised-learning method)
  and was best at lambda = 0 (the extreme TD method), as shown in Figure 3." Explanation: Widrow-Hoff "only
  minimizes error on the training set; it does not necessarily minimize error for future experience."
- Experiment 2 (p.22): "For all values, however, the Widrow-Hoff procedure, TD(1), produced the worst estimates. All
  of the TD methods with lambda < 1 performed better both in absolute terms and over a wider range of alpha values
  than did the supervised-learning method." "in this experiment the best lambda value was not 0, but somewhere near
  0.3." Reason (p.22): "TD(0) is relatively slow at propagating prediction levels back along a sequence."
- Conclusion (p.40): "TD methods appear to be computationally cheaper and to learn faster than conventional
  approaches to prediction learning." Abstract: "Here we prove their convergence and optimality for special cases";
  TD methods "require less memory and less peak computation than conventional methods and they produce more
  accurate predictions."

## 9. Limitations the source states
- Convergence only in the mean (expected value) for linear TD(0) and TD(1); p.29: "We conjecture that these
  stronger forms of convergence hold for linear TD(0) as well, but this remains an open question. Also open is the
  question of convergence of linear TD(lambda) for 0 < lambda < 1." Later fixed (per S&B ch. 6 remarks, p.139):
  TD(0) w.p. 1 by Dayan (1992); Jaakkola, Jordan, and Singh (1994) and Tsitsiklis (1994).
- Footnote 4 (p.23): Witten (1977) "presented a sketch of a convergence proof for a TD procedure ... but many steps
  were left out, and it now appears that the theorem he proposed is not true."
- The optimal (ML) predictions themselves are impractical: O(n^2) memory and up to O(n^3) computation (p.30).

## 10. Outline feedback (ch-mctd "Where this came from")
- Sutton 1988 is a prediction paper: no actions, no per-step rewards in the base setting. Present it as "TD as a
  general prediction method", not as a control algorithm (S&B p.21 says the same).
- The famous random-walk result is about lambda, not TD vs MC alone: TD(0) best under repeated presentation, lambda
  near 0.3 best after a single presentation, TD(1) = Widrow-Hoff worst in both. S&B's Example 6.2 curves are a
  related but different experiment (TD(0) vs constant-alpha MC learning curves, 100 episodes, 100 runs); S&B p.139
  credits the random walk example itself to Sutton (1988). (The p.158 remark, "made for this text", covers the
  chapter 7 random-walk results.)
- If the page quotes one sentence, use the abstract's credit-assignment sentence in section 4 above.

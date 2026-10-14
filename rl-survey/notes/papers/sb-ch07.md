# S&B 2nd ed., Chapter 7: n-step Bootstrapping

Source: `sources/sutton-barto/RLbook2020.pdf` (2020 printing). Book page = PDF page - 22. Equations, the box and
Example 7.1 were checked on rendered PDF pages; Figure 7.2 curve labels were read from a 400 dpi render.

## 1. Identity
- Sutton and Barto, 2nd ed. (2018), Chapter 7 "n-step Bootstrapping", book pp. 141-158 (PDF 163-180).
- Sections: 7.1 n-step TD Prediction p.142; 7.2 n-step Sarsa p.145; 7.3 n-step Off-policy Learning p.148; 7.4
  *Per-decision Methods with Control Variates p.150; 7.5 Off-policy Learning Without Importance Sampling: The
  n-step Tree Backup Algorithm p.152; 7.6 *A Unifying Algorithm: n-step Q(sigma) p.154; 7.7 Summary p.157;
  Bibliographical and Historical Remarks p.158.

## 2. Core problem
"Neither MC methods nor one-step TD methods are always the best" (p.141). Find the methods in between, and free the
bootstrapping interval from "the tyranny of the time step" (p.141).

## 3. What prior methods did and why that hurt
One-step TD uses the same time step for acting and for bootstrapping; "bootstrapping works best if it is over a
length of time in which a significant and recognizable state change has occurred" (p.141). MC waits for the whole
episode.

## 4. The one key idea
Use n real rewards, then bootstrap from V at step t+n. n = 1 is TD(0); n >= T - t is Monte Carlo. "The best methods
are often intermediate between the two extremes" (p.141).

## 5. Mechanism (box transcribed)

n-step TD for estimating V ~ v_pi (box, p.144, PDF 166):
```
Input: a policy pi
Algorithm parameters: step size alpha in (0, 1], a positive integer n
Initialize V(s) arbitrarily, for all s in S
All store and access operations (for S_t and R_t) can take their index mod n + 1
Loop for each episode:
    Initialize and store S_0 != terminal
    T <- infinity
    Loop for t = 0, 1, 2, ... :
    |   If t < T, then:
    |       Take an action according to pi(.|S_t)
    |       Observe and store the next reward as R_{t+1} and the next state as S_{t+1}
    |       If S_{t+1} is terminal, then T <- t + 1
    |   tau <- t - n + 1   (tau is the time whose state's estimate is being updated)
    |   If tau >= 0:
    |       G <- sum_{i=tau+1}^{min(tau+n, T)} gamma^{i-tau-1} R_i
    |       If tau + n < T, then: G <- G + gamma^n V(S_{tau+n})        (G_{tau:tau+n})
    |       V(S_tau) <- V(S_tau) + alpha [G - V(S_tau)]
    Until tau = T - 1
```
"no changes at all are made during the first n - 1 steps of each episode. To make up for that, an equal number of
additional updates are made at the end of the episode" (p.143).

## 6. Equations (numbers as printed)

One-step and two-step returns (unnumbered, p.143):
```tex
G_{t:t+1} \doteq R_{t+1} + \gamma V_t(S_{t+1}), \qquad G_{t:t+2} \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 V_{t+1}(S_{t+2})
```

Eq. 7.1, n-step return (p.143, PDF 165):
```tex
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})
```
"for all n, t such that n >= 1 and 0 <= t < T - n." "If t + n >= T ... the n-step return defined to be equal to the
ordinary full return (G_{t:t+n} = G_t if t + n >= T)." Book: "All n-step returns can be considered approximations
to the full return, truncated after n steps and then corrected for the remaining missing terms by
V_{t+n-1}(S_{t+n})." Gloss: n real discounted rewards, plus gamma^n times the current estimate of the state
reached after n steps; V_{t+n-1} = the value table as it stands when the update is made (time t+n).

Eq. 7.2, n-step TD (p.143):
```tex
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha\big[G_{t:t+n} - V_{t+n-1}(S_t)\big], \qquad 0 \le t < T
```
"while the values of all other states remain unchanged". Called "n-step TD".

Eq. 7.3, error reduction property (p.144):
```tex
\max_s \Big|\mathbb{E}_\pi[G_{t:t+n} \mid S_t = s] - v_\pi(s)\Big| \le \gamma^n \max_s \Big|V_{t+n-1}(s) - v_\pi(s)\Big|
```
for all n >= 1. Used to show n-step TD methods converge "under appropriate technical conditions".

Off-policy n-step (Section 7.3, p.148-149), relevant to the importance-sampling plant:
- Eq. 7.9: V_{t+n}(S_t) = V_{t+n-1}(S_t) + alpha rho_{t:t+n-1} [G_{t:t+n} - V_{t+n-1}(S_t)], 0 <= t < T.
- Eq. 7.10:
  ```tex
  \rho_{t:h} \doteq \prod_{k=t}^{\min(h, T-1)} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
  ```
  "(cf. Eq. 5.3)". "if the two policies are actually the same (the on-policy case) then the importance sampling
  ratio is always 1."
- Eq. 7.11 (off-policy n-step Sarsa): Q_{t+n}(S_t,A_t) = Q_{t+n-1}(S_t,A_t) + alpha rho_{t+1:t+n} [G_{t:t+n} -
  Q_{t+n-1}(S_t,A_t)]. "the importance sampling ratio here starts and ends one step later than for n-step TD (7.9).
  This is because here we are updating a state-action pair. We do not have to care how likely we were to select
  the action; now that we have selected it we want to learn fully from what happens, with importance sampling only
  for subsequent actions" (p.149).
- Section 7.5 (p.152): "Is off-policy learning possible without importance sampling? Q-learning and Expected Sarsa
  from Chapter 6 do this for the one-step case".

## 7. Example settings

Example 7.1 n-step TD Methods on the Random Walk (p.144-145, PDF 166-167):
- Warm-up on the 5-state walk of Example 6.2: if the first episode goes C -> D -> E -> right terminal (return 1),
  with all V = 0.5, a one-step method changes only V(E); two-step changes V(D) and V(E); n > 2 changes all three
  visited states "all by the same amount".
- Figure 7.2 task: "a larger random walk process, with 19 states instead of 5 (and with a -1 outcome on the left,
  all values initialized to 0)". Other rules carry over from Example 6.2 by reference (start in the center, left or
  right with equal probability, +1 on the right, undiscounted); the text does not restate them.
- Methods: n-step TD "with a range of values for n and alpha". Curve labels in Figure 7.2: n = 1, 2, 4, 8, 16, 32,
  64, 128, 256, 512. x-axis alpha from 0 to 1.
- Measure: "the square-root of the average squared error between the predictions at the end of the episode for the
  19 states and their true values, then averaged over the first 10 episodes and 100 repetitions of the whole
  experiment (the same sets of walks were used for all parameter settings)." y-axis label: "Average RMS error over
  19 states and first 10 episodes". Section 9.4 (p.209) restates it as "an unweighted average of the RMS error over
  all states and over the first 10 episodes".
- Finding (text): "Note that methods with an intermediate value of n worked best. This illustrates how the
  generalization of TD and Monte Carlo methods to n-step methods can potentially perform better than either of the
  two extreme methods." The text names no single best n; do not read one off the plot.
- Derived by us (not printed anywhere in the book): with -1/+1 outcomes and equal steps, true values of states 1..19
  are -0.9, -0.8, ..., 0.9 (state i: i/10 - 1).
- Exercise 7.3 (p.145) asks why 19 states and why the left outcome changed "from 0 to -1".

## 8. Results and findings
- Figure 7.2 (p.145): intermediate n best (text). Figure 12.3 (p.291) repeats Figure 7.2 alongside the offline
  lambda-return: "In both case, intermediate values of the bootstrapping parameter (lambda or n) performed best."
- Summary (p.157): "Methods that involve an intermediate amount of bootstrapping are important because they will
  typically perform better than either extreme." Costs: n-step methods "involve a delay of n time steps before
  updating", more computation per step, and more memory.
- Off-policy summary (p.157): importance sampling "is conceptually simple but can be of high variance"; tree backup
  "involves no importance sampling".

## 9. Limitations and history (remarks p.158, PDF 180)
- "The notion of n-step returns is due to Watkins (1989), who also first discussed their error reduction property."
- n-step methods were "treated as of conceptual interest, but not feasible in practice" in the first edition;
  Cichosz (1995) and van Seijen (2016) "showed that they are actually completely practical algorithms."
- 7.1-2: "The results in the random walk examples were made for this text based on work of Sutton (1988) and Singh
  and Sutton (1996)."
- Tree backup: Precup, Sutton, and Singh (2000). Q(sigma): new to this text; related work De Asis et al. (2017).

## 10. Outline feedback (ch-mctd viz-nstep, ch-qlearn)
- viz-nstep: the book gives the full protocol (above). The "readout of the best n" must come from our computed run;
  the book only says "intermediate". The book's n grid is powers of two from 1 to 512; alpha spans 0 to 1.
- The 19-state true values are needed to compute the RMS error but are not printed; mark them as derived on the
  page (or derive them in a sentence).
- "Equations: ... n-step return (S&B Eq. 7.1)" is correct; the update that uses it is Eq. 7.2.
- ch-qlearn "Why one-step Q-learning needs no ratio": the cleanest sourced wording is the Eq. 7.11 passage (p.149)
  plus the Section 7.5 sentence (p.152). For one-step action values, the ratio would cover only the next action,
  and Q-learning / Expected Sarsa replace that sampled next action with the target policy's max / expectation.
- GAE plant (chapter 11): S&B's own bridge is "n-step returns -> lambda-return (Chapter 12)"; Figure 12.3 (p.291)
  overlays n-step and lambda-return on this same 19-state walk.

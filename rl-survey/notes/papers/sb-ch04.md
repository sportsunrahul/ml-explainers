# S&B Chapter 4: Dynamic Programming

## 1. Identity
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018; 2020 printing read). See
  sb-ch01.md for the full book identity. Book page = PDF page - 22.
- Chapter 4 "Dynamic Programming", book pp. 73-90 (PDF 95-112).
- Sections: 4.1 Policy Evaluation (Prediction) p.74; 4.2 Policy Improvement p.76; 4.3 Policy Iteration p.80;
  4.4 Value Iteration p.82; 4.5 Asynchronous Dynamic Programming p.85; 4.6 Generalized Policy Iteration p.86;
  4.7 Efficiency of Dynamic Programming p.87; 4.8 Summary p.88; Bibliographical and Historical Remarks p.89.

## 2. Core problem
Compute v_pi for a given policy and an optimal policy, given "a perfect model of the environment as a Markov
decision process" (p.73). The state, action and reward sets are finite and p(s', r|s, a) is known.

## 3. What prior methods did and why that hurt (per the source)
- Solving the |S| linear equations (4.4) directly is "a straightforward, if tedious, computation"; the book uses
  iterative methods instead (p.74).
- Direct search in policy space must examine all k^n deterministic policies; DP is polynomial in n and k (4.7,
  p.87).
- Linear programming "become[s] impractical at a much smaller number of states than do DP methods (by a factor of
  about 100)" (p.87).
- Classical DP itself is "of limited utility in reinforcement learning both because of their assumption of a
  perfect model and because of their great computational expense" (p.73).

## 4. The one key idea
"DP algorithms are obtained by turning Bellman equations such as these into assignments, that is, into update
rules for improving approximations of the desired value functions." (p.74)

## 5. Algorithms, step by step (boxes transcribed from the rendered PDF)
Box "Iterative Policy Evaluation, for estimating V ~ v_pi" (4.1, p.75, PDF 97):
```
Input pi, the policy to be evaluated
Algorithm parameter: a small threshold theta > 0 determining accuracy of estimation
Initialize V(s) arbitrarily, for s in S, and V(terminal) to 0
Loop:
    Delta <- 0
    Loop for each s in S:
        v <- V(s)
        V(s) <- sum_a pi(a|s) sum_{s',r} p(s',r|s,a) [r + gamma V(s')]
        Delta <- max(Delta, |v - V(s)|)
until Delta < theta
```
This box is the in-place version (one array). The two-array version keeps v_k and v_{k+1} separately (p.75).

Box "Policy Iteration (using iterative policy evaluation) for estimating pi ~ pi_*" (4.3, p.80, PDF 102):
```
1. Initialization
   V(s) in R and pi(s) in A(s) arbitrarily for all s in S; V(terminal) = 0   (the "=" carries a definition dot)
2. Policy Evaluation
   Loop:
       Delta <- 0
       Loop for each s in S:
           v <- V(s)
           V(s) <- sum_{s',r} p(s',r|s,pi(s)) [r + gamma V(s')]
           Delta <- max(Delta, |v - V(s)|)
   until Delta < theta (a small positive number determining the accuracy of estimation)
3. Policy Improvement
   policy-stable <- true
   For each s in S:
       old-action <- pi(s)
       pi(s) <- argmax_a sum_{s',r} p(s',r|s,a) [r + gamma V(s')]
       If old-action != pi(s), then policy-stable <- false
   If policy-stable, then stop and return V ~ v_* and pi ~ pi_*; else go to 2
```
Exercise 4.4 (p.82): "The policy iteration algorithm on page 80 has a subtle bug in that it may never terminate if
the policy continually switches between two or more policies that are equally good."

Box "Value Iteration, for estimating pi ~ pi_*" (4.4, p.83, PDF 105):
```
Algorithm parameter: a small threshold theta > 0 determining accuracy of estimation
Initialize V(s), for all s in S+, arbitrarily except that V(terminal) = 0
Loop:
    Delta <- 0
    Loop for each s in S:
        v <- V(s)
        V(s) <- max_a sum_{s',r} p(s',r|s,a) [r + gamma V(s')]
        Delta <- max(Delta, |v - V(s)|)
until Delta < theta
Output a deterministic policy, pi ~ pi_*, such that
    pi(s) = argmax_a sum_{s',r} p(s',r|s,a) [r + gamma V(s')]
```
Note the init sets differ: the PE box initializes over S plus V(terminal) = 0; the VI box over S+.

Policy iteration sequence (p.80): `\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*`,
E = policy evaluation, I = policy improvement. "Because a finite MDP has only a finite number of deterministic
policies, this process must converge to an optimal policy and the optimal value function in a finite number of
iterations."

## 6. Equations (transcribed from the PDF; all verified on the rendered pages)
- Eq. 4.1 (p.73, PDF 95): `v_*(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] = \max_a \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_*(s')\big]`
- Eq. 4.2 (p.73): `q_*(s, a) = \mathbb{E}\big[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a\big] = \sum_{s', r} p(s', r|s, a)\big[r + \gamma \max_{a'} q_*(s', a')\big]`
  (4.1 and 4.2 restate 3.19 and 3.20.)
- Eqs. 4.3-4.4 (4.1, p.74, PDF 96):
  `v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]` (4.3)
  `= \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_\pi(s')\big]` (4.4)
  "The existence and uniqueness of v_pi are guaranteed as long as either gamma < 1 or eventual termination is
  guaranteed from all states under the policy pi."
- Eq. 4.5 (p.74), iterative policy evaluation:
  `v_{k+1}(s) \doteq \mathbb{E}_\pi[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s] = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_k(s')\big]`,
  for all s in S. "Clearly, v_k = v_pi is a fixed point for this update rule". v_0 is arbitrary "except that the
  terminal state, if any, must be given value 0". "We call this kind of operation an expected update." Gloss:
  v_k = estimate after k sweeps; the right side uses only old values v_k (two-array form).
- Eq. 4.6 (4.2, p.78, PDF 100): `q_\pi(s, a) \doteq \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] = \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_\pi(s')\big]`
- Eq. 4.7 (p.78): `q_\pi(s, \pi'(s)) \ge v_\pi(s)`, for all s in S (pi, pi' deterministic).
- Eq. 4.8 (p.78): `v_{\pi'}(s) \ge v_\pi(s)`, for all s in S.
  Policy improvement theorem: if (4.7) holds then (4.8) holds; "Moreover, if there is strict inequality of (4.7)
  at any state, then there must be strict inequality of (4.8) at that state." Proof sketch (p.78): expand q_pi with
  (4.6) and reapply (4.7) repeatedly until reaching v_pi'(s).
- Eq. 4.9 (p.79, PDF 101), greedy policy:
  `\pi'(s) \doteq \operatorname*{argmax}_a q_\pi(s, a) = \operatorname*{argmax}_a \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] = \operatorname*{argmax}_a \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_\pi(s')\big]`
  "with ties broken arbitrarily". "The greedy policy takes the action that looks best in the short term--after one
  step of lookahead--according to v_pi." If pi' is as good as but not better than pi, then v_pi' satisfies the
  Bellman optimality equation (4.1), so both are optimal: "Policy improvement thus must give us a strictly better
  policy except when the original policy is already optimal." (p.79) Stochastic case: ties may share probability;
  "Any apportioning scheme is allowed as long as all submaximal actions are given zero probability." (p.79)
- Eq. 4.10 (4.4, p.83, PDF 105), value iteration:
  `v_{k+1}(s) \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s, A_t = a] = \max_a \sum_{s', r} p(s', r|s, a)\big[r + \gamma v_k(s')\big]`,
  for all s in S. "value iteration is obtained simply by turning the Bellman optimality equation into an update
  rule. Also note how the value iteration update is identical to the policy evaluation update (4.5) except that it
  requires the maximum to be taken over all actions." It "combines the policy improvement and truncated policy
  evaluation steps" (policy evaluation "stopped after just one sweep").

## 7. Config / example settings
Example 4.1, 4x4 gridworld (4.1, p.76, PDF 98):

| Setting | Value |
|---|---|
| Layout | 4x4 grid; terminal state shaded at top-left and bottom-right ("although it is shown in two places, it is formally one state") |
| Nonterminal states | S = {1, 2, ..., 14}, numbered row by row from the cell right of the top-left terminal |
| Actions | A = {up, down, right, left}, deterministic |
| Off-grid | "actions that would take the agent off the grid in fact leave the state unchanged" |
| Examples | p(6, -1 \| 5, right) = 1, p(7, -1 \| 7, right) = 1, p(10, r \| 5, right) = 0 for all r |
| Reward | -1 on all transitions until the terminal state is reached; r(s, a, s') = -1 |
| Discount | "This is an undiscounted, episodic task" (gamma = 1) |
| Policy | equiprobable random (all actions equally likely) |
| Meaning of v_pi | "the negation of the expected number of steps from that state until termination" |

Grid numbering (Figure in Example 4.1, p.76):
```
 T   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14   T
```

Figure 4.1 (p.77, PDF 99), left column, v_k for the random policy. The figure shows six panels: k = 0, 1, 2, 3,
10, and infinity. Numbers "are rounded to two significant digits" (caption). As printed:
```
k = 0:  0.0  0.0  0.0  0.0 /  0.0  0.0  0.0  0.0 /  0.0  0.0  0.0  0.0 /  0.0  0.0  0.0  0.0
k = 1:  0.0 -1.0 -1.0 -1.0 / -1.0 -1.0 -1.0 -1.0 / -1.0 -1.0 -1.0 -1.0 / -1.0 -1.0 -1.0  0.0
k = 2:  0.0 -1.7 -2.0 -2.0 / -1.7 -2.0 -2.0 -2.0 / -2.0 -2.0 -2.0 -1.7 / -2.0 -2.0 -1.7  0.0
k = 3:  0.0 -2.4 -2.9 -3.0 / -2.4 -2.9 -3.0 -2.9 / -2.9 -3.0 -2.9 -2.4 / -3.0 -2.9 -2.4  0.0
k = 10: 0.0 -6.1 -8.4 -9.0 / -6.1 -7.7 -8.4 -8.4 / -8.4 -8.4 -7.7 -6.1 / -9.0 -8.4 -6.1  0.0
k = inf:0.0 -14. -20. -22. / -14. -18. -20. -20. / -20. -20. -18. -14. / -22. -20. -14.  0.0
```
(rows of the grid separated by "/"; the k = infinity panel prints "-14." etc. with a trailing point.)
Right column: greedy policy w.r.t. v_k; k = 0 is labelled "random policy"; k = 3, 10 and infinity are labelled
"optimal policy".

Our verification (not printed in the book): a script of the two-array update (Eq. 4.5, all states from v_k, the
sequence the Example 4.1 text says the figure shows) reproduces every printed row; the in-place box does not
(in-place k = 1 already gives -1.25 for state 2). The
exact two-array values behind the printed numbers include -1.75 (printed -1.7), -2.4375 (-2.4), -2.875 and -2.9375
(-2.9), -6.138 (-6.1), -7.737 (-7.7), -8.352 and -8.428 (-8.4), -8.967 (-9.0). The limit is the integers -14, -18,
-20, -22. Sweeps to Delta < 0.01: 89 two-array, 62 in-place (our run).

Other DP examples (not needed by the outline): Example 4.2 Jack's Car Rental (p.81; gamma = 0.9, Poisson
lambdas 3, 4 requests and 3, 2 returns, max 20 cars, max 5 moved, $10 rent, $2 per car moved); Example 4.3
Gambler's Problem (p.84; p_h = 0.4, goal $100, value iteration sweeps shown).

## 8. Results and findings worth remembering
Convergence and speed (all the chapter says; it gives no gamma-contraction rate):
- "the sequence {v_k} can be shown in general to converge to v_pi as k -> infinity under the same conditions that
  guarantee the existence of v_pi." (p.74)
- In-place: "This in-place algorithm also converges to v_pi; in fact, it usually converges faster than the
  two-array version, as you might expect, because it uses new data as soon as they are available. ... For the
  in-place algorithm, the order in which states have their values updated during the sweep has a significant
  influence on the rate of convergence. We usually have the in-place version in mind when we think of DP
  algorithms." (p.75)
- "Formally, iterative policy evaluation converges only in the limit, but in practice it must be halted short of
  this. The pseudocode tests the quantity max_s |v_{k+1}(s) - v_k(s)| after each sweep and stops when it is
  sufficiently small." (p.75)
- Policy iteration warm start: "each policy evaluation, itself an iterative computation, is started with the value
  function for the previous policy. This typically results in a great increase in the speed of convergence of
  policy evaluation (presumably because the value function changes little from one policy to the next)." (p.80)
- "Policy iteration often converges in surprisingly few iterations" (p.82). For Example 4.1, "policy iteration
  would find the optimal policy after just one iteration." (p.82)
- Value iteration: "For arbitrary v_0, the sequence {v_k} can be shown to converge to v_* under the same conditions
  that guarantee the existence of v_*." (p.83) "Like policy evaluation, value iteration formally requires an
  infinite number of iterations to converge exactly to v_*. In practice, we stop once the value function changes
  by only a small amount in a sweep." (p.83)
- Truncated policy iteration: "Faster convergence is often achieved by interposing multiple policy evaluation
  sweeps between each policy improvement sweep." "All of these algorithms converge to an optimal policy for
  discounted finite MDPs." (pp.83-84)
- Asynchronous VI: "If 0 <= gamma < 1, asymptotic convergence to v_* is guaranteed given only that all states occur
  in the sequence {s_k} an infinite number of times" (p.85).
- Efficiency: "in the worst case, the time that DP methods take to find an optimal policy is polynomial in the
  number of states and actions ... even though the total number of (deterministic) policies is k^n. In this sense,
  DP is exponentially faster than any direct search in policy space could be" (p.87). "In practice, these methods
  usually converge much faster than their theoretical worst-case run times, particularly if they are started with
  good initial value functions or policies." (p.87)
Policy settles before values:
- Figure 4.1 caption: "The last policy is guaranteed only to be an improvement over the random policy, but in this
  case it, and all policies after the third iteration, are optimal."
- p.82: "In that example, policy evaluation iterations beyond the first three have no effect on the corresponding
  greedy policy."
- p.79: for the greedy pi' on the last row, v_pi'(s) is "either -1, -2, or -3, for all states", "whereas v_pi(s) is
  at most -14".
GPI (4.6, p.86, PDF 108):
- "We use the term generalized policy iteration (GPI) to refer to the general idea of letting policy-evaluation
  and policy-improvement processes interact, independent of the granularity and other details of the two
  processes. Almost all reinforcement learning methods are well described as GPI."
- "If both the evaluation process and the improvement process stabilize, that is, no longer produce changes, then
  the value function and policy must be optimal." "The evaluation and improvement processes in GPI can be viewed
  as both competing and cooperating." (pp.86-87)
Efficiency and the curse of dimensionality (4.7, p.87, PDF 109):
- "DP may not be practical for very large problems, but compared with other methods for solving MDPs, DP methods
  are actually quite efficient."
- "DP is sometimes thought to be of limited applicability because of the curse of dimensionality, the fact that
  the number of states often grows exponentially with the number of state variables. Large state sets do create
  difficulties, but these are inherent difficulties of the problem, not of DP as a solution method. In fact, DP is
  comparatively better suited to handling large state spaces than competing methods such as direct search and
  linear programming."
- "In practice, DP methods can be used with today's computers to solve MDPs with millions of states. Both policy
  iteration and value iteration are widely used, and it is not clear which, if either, is better in general."
- Sweeps cost (4.5, p.85): "the game of backgammon has over 10^20 states. Even if we could perform the value
  iteration update on a million states per second, it would take over a thousand years to complete a single
  sweep."
Bootstrapping (4.8, p.89): DP methods "update estimates on the basis of other estimates. We call this general idea
bootstrapping."
History (bib, pp.89-90): "The term 'dynamic programming' is due to Bellman (1957a)"; "The policy improvement
theorem and the policy iteration algorithm are due to Bellman (1957a) and Howard (1960)"; value iteration as
truncated policy iteration follows Puterman and Shin (1978), "modified policy iteration"; Watkins (1989)
characterized RL methods as "incremental dynamic programming"; first DP-RL connection by Minsky (1961) on Samuel's
checkers player; two-array = "Jacobi-style" or "synchronous", in-place = "Gauss-Seidel-style"; asynchronous DP
due to Bertsekas (1982, 1983); "The phrase 'curse of dimensionality' is due to Bellman (1957a)" (p.90).

## 9. Limitations the source states
- Needs a complete and accurate model and sweeps over the entire state set (pp.73, 85).
- Sweeps can be "prohibitively expensive" for large state sets; asynchronous DP and GPI variants help (p.85, p.88).
- PI box may cycle between equally good policies (Exercise 4.4, p.82).
- Undiscounted episodic asynchronous updates: "it is possible that there are some orderings of updates that do not
  result in convergence, but it is relatively easy to avoid these" (footnote 1, p.85).
- Later fixes named by the source: Monte Carlo (no model, no bootstrap, ch. 5) and TD (no model, bootstraps,
  ch. 6) (p.89).

## 10. Outline feedback (ch-dp)
1. Equation numbers are right: iterative policy evaluation Eq. 4.5 (p.74), greedy improvement Eq. 4.9 (p.79),
   value iteration Eq. 4.10 (p.83); policy improvement theorem Eqs. 4.7-4.8 (p.78). Add Eq. 4.6 (q_pi from v_pi,
   p.78), which ch-bellman also needs.
2. Reproducing Figure 4.1 needs the two-array (synchronous) update. Example 4.1 says the figure shows "the
   sequence of value functions {v_k} computed by iterative policy evaluation", and Eq. 4.5 defines v_{k+1} from
   v_k for all s, which is the two-array sequence; the box (p.75) is the in-place implementation. Our computation
   confirms it: the printed k = 1, 2, 3, 10 rows match two-array sweeps, not the in-place box. Only k = infinity
   matches both. So either code the NumPy check with two arrays (and say why it differs from the box), or check
   only the k = infinity row. Compare with a tolerance: the figure rounds to two significant digits and prints
   -1.75 as -1.7.
3. "Each sweep shrinks the error by about a factor gamma (contraction)" is not in S&B ch. 1-4. The book states
   only convergence in the limit, under "gamma < 1 or eventual termination" (pp.74-75), and qualitative speed
   remarks (pp.75, 80, 82, 87). Contraction appears only in the bib remarks (Denardo 1967, "contraction mappings",
   p.71) and as the Bellman operator in ch. 11 (Eqs. 11.20-11.21, p.269), with no rate. Derive the factor-gamma
   bound on the page and label it as the page's derivation. Run the log plot on the yard with gamma < 1: Example
   4.1 has gamma = 1, where no factor-gamma shrink exists.
4. "The policy settles long before the values do": cite the Figure 4.1 caption ("all policies after the third
   iteration, are optimal") and p.82 ("iterations beyond the first three have no effect on the corresponding
   greedy policy").
5. Policy iteration on the yard: Exercise 4.4 says the PI box may never stop if it switches between equally good
   policies. Break ties in a fixed order, or stop when the values stop changing; slip noise makes ties likely.
6. GPI and "the catch": quote 4.6 (p.86) for GPI; for the catch use p.73 ("assumption of a perfect model and ...
   great computational expense") and the backgammon sweep cost (p.85). The curse of dimensionality in 4.7 is
   framed as a property of the problem, not of DP (p.87); keep that nuance if the page uses the phrase.
7. "Where this came from": the term DP and the curse of dimensionality are Bellman (1957a); policy iteration is
   Bellman (1957a) and Howard (1960); value iteration as truncated PI follows Puterman and Shin (1978); Watkins
   (1989) called RL "incremental dynamic programming" (pp.89-90).
8. Self-check "after k sweeps, how far has information travelled?": Figure 4.1 shows it for the two-array update.
   At k = 2 only the cells next to a terminal differ (-1.7 vs -2.0); at k = 3 the cells two steps away differ too
   (-2.9 vs -3.0). So the terminal's influence moves one cell per sweep (our reading of the printed rows). For
   in-place sweeps the book says only that state order "has a significant influence on the rate of convergence"
   (p.75). State which update the viz uses.

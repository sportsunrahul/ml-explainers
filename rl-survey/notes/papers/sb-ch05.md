# S&B 2nd ed., Chapter 5: Monte Carlo Methods

Source: `sources/sutton-barto/RLbook2020.pdf` (2020 printing). Book page = PDF page - 22. Equations and boxes
below were checked against the rendered PDF pages, not the text dump (the dump drops minus signs and gamma, and
maps alpha/pi/rho/epsilon to junk glyphs).

## 1. Identity
- Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), Chapter 5 "Monte Carlo Methods",
  book pp. 91-118 (PDF 113-140).
- Sections: 5.1 Monte Carlo Prediction p.92; 5.2 Monte Carlo Estimation of Action Values p.96; 5.3 Monte Carlo
  Control p.97; 5.4 Monte Carlo Control without Exploring Starts p.100; 5.5 Off-policy Prediction via Importance
  Sampling p.103; 5.6 Incremental Implementation p.109; 5.7 Off-policy Monte Carlo Control p.110; 5.8
  *Discounting-aware Importance Sampling p.112; 5.9 *Per-decision Importance Sampling p.114; 5.10 Summary p.115;
  Bibliographical and Historical Remarks p.116.

## 2. Core problem
Estimate v_pi and q_pi, and find good policies, from sample episodes alone, with no model of the dynamics. MC is
defined only for episodic tasks: "Only on the completion of an episode are value estimates and policies changed"
(p.91). It is incremental episode by episode, not step by step.

## 3. What prior methods did and why that hurt
DP needs the full distribution p(s', r | s, a). Blackjack: "it would not be easy to apply DP methods ... All of the
probabilities must be computed before DP can be applied, and such computations are often complex and error-prone.
In contrast, generating the sample games required by Monte Carlo methods is easy" (p.94).

## 4. The one key idea
The value of a state is the expected return, so average the returns observed after visits to it (p.92). Off-policy:
reweight each return by the importance-sampling ratio of target to behavior policy (Section 5.5).

## 5. Mechanism (boxes transcribed)

First-visit MC prediction, for estimating V ~ v_pi (box, p.92, PDF 114):
```
Input: a policy pi to be evaluated
Initialize:
    V(s) in R, arbitrarily, for all s in S
    Returns(s) <- an empty list, for all s in S
Loop forever (for each episode):
    Generate an episode following pi: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T
    G <- 0
    Loop for each step of episode, t = T-1, T-2, ..., 0:
        G <- gamma*G + R_{t+1}
        Unless S_t appears in S_0, S_1, ..., S_{t-1}:
            Append G to Returns(S_t)
            V(S_t) <- average(Returns(S_t))
```
"Every-visit MC would be the same except without the check for S_t having occurred earlier in the episode" (p.92).

Off-policy MC prediction (policy evaluation) for estimating Q ~ q_pi (box, p.110, PDF 132): weighted IS, done
incrementally, backward through the episode:
```
Input: an arbitrary target policy pi
Initialize, for all s in S, a in A(s):
    Q(s,a) in R (arbitrarily)
    C(s,a) <- 0
Loop forever (for each episode):
    b <- any policy with coverage of pi
    Generate an episode following b: S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T
    G <- 0
    W <- 1
    Loop for each step of episode, t = T-1, T-2, ..., 0, while W != 0:
        G <- gamma*G + R_{t+1}
        C(S_t,A_t) <- C(S_t,A_t) + W
        Q(S_t,A_t) <- Q(S_t,A_t) + (W / C(S_t,A_t)) [G - Q(S_t,A_t)]
        W <- W * pi(A_t|S_t) / b(A_t|S_t)
```

Terms (Section 5.5, p.103): target policy = "the policy being learned about"; behavior policy = "the policy used
to generate behavior"; off-policy learning = learning "from data 'off' the target policy". Coverage (p.103):
"pi(a|s) > 0 implies b(a|s) > 0". On-policy vs off-policy (p.100): "On-policy methods attempt to evaluate or
improve the policy that is used to make decisions, whereas off-policy methods evaluate or improve a policy
different from that used to generate the data."

## 6. Equations (printed numbers verified on the PDF page images)

Eq. 5.3, importance-sampling ratio (Section 5.5, p.104, PDF 126):
```tex
\rho_{t:T-1} \doteq \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)\, p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)\, p(S_{k+1}|S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
```
Book wording: "the relative probability of the trajectory under the target and behavior policies (the
importance-sampling ratio)"; the transition probabilities "appear identically in both the numerator and
denominator, and thus cancel. The importance sampling ratio ends up depending only on the two policies and the
sequence, not on the MDP." Gloss: rho = product over the remaining steps of (prob. the target policy pi takes the
logged action) / (prob. the behavior policy b took it); p = environment transition probability (cancels).

Eq. 5.4 (p.104): the ratio fixes the expectation of behavior returns.
```tex
\mathbb{E}[\rho_{t:T-1} G_t \mid S_t = s] = v_\pi(s)
```
Context: returns from b "have the wrong expectation E[G_t | S_t = s] = v_b(s)".

Eq. 5.5, ordinary importance sampling (p.104):
```tex
V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}
```
T(s) = set of time steps at which s is visited (time steps numbered across episode boundaries; first visits only
for a first-visit method); T(t) = first termination after t; G_t = return after t up through T(t). "When
importance sampling is done as a simple average in this way it is called ordinary importance sampling" (p.105).

Eq. 5.6, weighted importance sampling (p.105, PDF 127):
```tex
V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}
```
"or zero if the denominator is zero."

Eqs. 5.7-5.8, incremental weighted average (Section 5.6, p.109): V_n = sum_{k=1}^{n-1} W_k G_k / sum_{k=1}^{n-1}
W_k (n >= 2); V_{n+1} = V_n + (W_n / C_n)[G_n - V_n] (n >= 1); C_{n+1} = C_n + W_{n+1}, C_0 = 0.

## 7. Example settings

Example 5.1 Blackjack (p.93): episodic, gamma = 1; rewards +1 win, -1 lose, 0 draw, all intermediate rewards 0;
actions hit/stick; infinite deck (with replacement); dealer sticks on 17 or more, else hits; state = player sum
(12-21), dealer showing card (ace-10), usable ace or not: "a total of 200 states". Policy evaluated: stick only on
20 or 21. Figure 5.1 (p.94): after 10,000 and 500,000 episodes; "after 500,000 games the value function is very
well approximated" (p.93).

Example 5.4 off-policy blackjack state (p.106): state = dealer showing deuce, player sum 13, usable ace. Behavior:
hit or stick at random with equal probability. Target: stick only on 20 or 21. True value "approximately -0.27726"
(from one hundred million target-policy episodes). 100 independent runs, each from estimates of zero, 10,000
episodes each; Figure 5.3 plots mean square error averaged over the 100 runs.

Example 5.5 Infinite variance (p.106-108, Figure 5.4 p.107): one nonterminal state s, actions left and right.
right -> terminates deterministically, reward 0. left -> back to s with probability 0.9, or to termination with
probability 0.1 and reward +1. Target policy pi(left|s) = 1; behavior b(left|s) = 1/2. True value v_pi(s) = 1
(gamma = 1). Figure 5.4: ten independent runs of first-visit MC with ordinary IS. Expected squared scaled return
= 0.1 sum_k 0.9^k 2^k 2 = 0.2 sum_k 1.8^k = infinity (p.108).

## 8. Results and findings
- First-visit MC: returns are i.i.d. unbiased estimates; "the standard deviation of its error falls as 1/sqrt(n),
  where n is the number of returns averaged" (p.93). Every-visit MC also converges (Singh and Sutton 1996).
- Off-policy variance, Section 5.5 p.103: "because the data is due to a different policy, off-policy methods are
  often of greater variance and are slower to converge."
- Ordinary vs weighted IS, p.105: "Ordinary importance sampling is unbiased whereas weighted importance sampling
  is biased (though the bias converges asymptotically to zero). On the other hand, the variance of ordinary
  importance sampling is in general unbounded because the variance of the ratios can be unbounded, whereas in the
  weighted estimator the largest weight on any single return is one." ... "In practice, the weighted estimator
  usually has dramatically lower variance and is strongly preferred."
- Example of extreme ordinary IS (p.105): ratio ten -> ordinary estimate is ten times the observed return.
- Every-visit versions of both are biased, bias falls to zero (p.105).
- Example 5.4 / Figure 5.3 (p.106): "the weighted importance-sampling method has much lower error at the
  beginning, as is typical in practice."
- Example 5.5 / Figure 5.4 (p.107): ordinary IS "Even after millions of episodes, the estimates fail to converge
  to the correct value of 1"; weighted IS gives exactly 1 after the first episode that ended with left.
- Summary (p.116): "Ordinary importance sampling produces unbiased estimates, but has larger, possibly infinite,
  variance, whereas weighted importance sampling always has finite variance and is preferred in practice."

## 9. Limitations the source states
- MC ES convergence "has not yet been formally proved" (p.99; partial solution Tsitsiklis 2002).
- Off-policy MC control "learns only from the tails of episodes" when nongreedy actions are common; the fix the
  book points to is TD learning (next chapter) (p.111).
- "Despite their conceptual simplicity, off-policy Monte Carlo methods for both prediction and control remain
  unsettled and are a subject of ongoing research" (p.116).
- History (remarks p.116-117): term "Monte Carlo" from the 1940s (Los Alamos); first-visit vs every-visit
  distinguished by Singh and Sutton (1996); MC ES introduced in the 1998 edition; per-decision IS by Precup, Sutton
  and Singh (2000); weighted IS also called normalized IS.

## 10. Outline feedback (ch-qlearn viz-ratio, ch-mctd)
- viz-ratio: the book's lesson is the contrast ordinary (unbiased, variance can be infinite) vs weighted (biased,
  bounded weights, preferred). Showing only one estimator converging hides that. Suggest two traces: ordinary and
  weighted. Example 5.5 is a ready, sourced two-action world for the slider (true value 1, b = 0.5, infinite
  variance under ordinary IS).
- Phrase "converges to pi's value" with care: ordinary IS is unbiased in expectation, yet its estimates can fail to
  converge in practice when variance is infinite (Example 5.5).
- The ratio the page should name is Eq. 5.3 (rho_{t:T-1}); the PPO plant uses the one-step factor
  pi(A_t|S_t)/b(A_t|S_t), which is rho_{t:t} in the book's notation (Eq. 7.10 defines rho_{t:h} in general).

# S&B ch. 12: Eligibility Traces (lambda-return only)

## 1. Identity
- Sutton and Barto, 2nd ed. (2018), RLbook2020.pdf (2020 printing). Book page = PDF page - 22. Chapter 12 starts
  p.287 (PDF 309).
- Sections (TOC): 12.1 The lambda-return 288; 12.2 TD(lambda) 292; 12.3 n-step Truncated lambda-return Methods 295;
  12.4 Online lambda-return 297; 12.5 True Online TD(lambda) 299; 12.6 Dutch Traces in MC 301; 12.7 Sarsa(lambda)
  303; 12.8 Variable lambda and gamma 307; 12.9 Off-policy Traces with Control Variates 309; 12.10 Watkins's
  Q(lambda) to Tree-Backup(lambda) 312; 12.11 Stable Off-policy Methods with Traces 314; 12.12 Implementation Issues
  316; 12.13 Conclusions 317. Bib remarks start p.319 (PDF 341).
- Scope here: Sec. 12.1 only (p.288-290, PDF 310-312), as the n-step dial behind GAE (ch-ac) and the lambda = 0 / 1
  special cases.

## 2. Core problem
Each n-step return is a valid target, but which n? Averaging all n-step returns with geometric weights gives one
target with a single knob, lambda, that slides between one-step TD and Monte Carlo.

## 3. What prior methods did and why that hurt
n-step methods (ch. 7) fix one n. A compound update can average any n-step returns with positive weights summing to
1 (p.288), but it can only be done "when the longest of its component updates is complete" (p.289).

## 4. The one key idea
Weight the n-step return by (1 - lambda) lambda^{n-1}. The weights sum to 1 and fade by lambda each step.

## 5. Mechanism, step by step
1. n-step return with function approximation: Eq. 12.1.
2. lambda-return: geometric average of all n-step returns, Eq. 12.2.
3. Episodic form: all n-step returns past termination equal G_t, so the tail collapses into one term with weight
   lambda^{T-t-1}, Eq. 12.3.
4. lambda = 1 -> Monte Carlo (G_t); lambda = 0 -> one-step return G_{t:t+1} (p.290).
5. Off-line lambda-return algorithm: semi-gradient update toward G^lambda_t at episode end, Eq. 12.4.

## 6. Equations (verified on the PDF page images)
- Eq. 12.1, Sec. 12.1, p.288 (PDF 310):
  `G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n\hat v(S_{t+n},\mathbf{w}_{t+n-1}),\quad 0 \le t \le T-n`
- Eq. 12.2, p.289 (PDF 311):
  `G_t^\lambda \doteq (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}`
  Source (p.289): "This average contains all the n-step updates, each weighted proportionally to lambda^{n-1} (where
  lambda in [0, 1)), and is normalized by a factor of 1 - lambda to ensure that the weights sum to 1". Note the
  printed range for Eq. 12.2 is lambda in [0, 1).
- Weights (p.289): "The one-step return is given the largest weight, 1 - lambda; the two-step return is given the
  next largest weight, (1 - lambda) lambda; the three-step return is given the weight (1 - lambda) lambda^2; and so
  on. The weight fades by lambda with each additional step."
- Eq. 12.3, p.290 (PDF 312):
  `G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n} + \lambda^{T-t-1}G_t`
  Source (p.290): "This equation makes it clearer what happens when lambda = 1. In this case the main sum goes to zero,
  and the remaining term reduces to the conventional return. Thus, for lambda = 1, updating according to the
  lambda-return is a Monte Carlo algorithm. On the other hand, if lambda = 0, then the lambda-return reduces to
  G_{t:t+1}, the one-step return. Thus, for lambda = 0, updating according to the lambda-return is a one-step TD
  method."
- Eq. 12.4, p.290: `\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\Big[G_t^\lambda - \hat v(S_t,\mathbf{w}_t)\Big]\nabla\hat v(S_t,\mathbf{w}_t),\quad t = 0,\ldots,T-1`
  (off-line lambda-return algorithm).

## 7. Settings
None needed (no experiment used by the page from this section).

## 8. Findings worth remembering
- Figure 12.1 caption (p.289): "If lambda = 0, then the overall update reduces to its first component, the one-step TD
  update, whereas if lambda = 1, then the overall update reduces to its last component, the Monte Carlo update."
- Figure 12.2 (p.290): weighting diagram; "total area = 1"; weight on the actual final return is lambda^{T-t-1}.
- Exercise 12.2 (p.290) asks for the half-life tau_lambda in terms of lambda (not answered in the text).
- History (bib remarks p.319, PDF 341): "The lambda-return and its error-reduction properties were introduced by
  Watkins (1989) and further developed by Jaakkola, Jordan, and Singh (1994)." "Compound updates were called 'complex
  backups' in the first edition of this book." The terms "forward view" and "backward view" are new to this text.
- Actor-critic link (Sec. 13.5, p.332): "The one-step return in (13.12) is merely replaced by G_{t:t+n} or
  G^lambda_t respectively."

## 9. Limitations stated
A compound (forward-view) update is delayed until its longest component is complete (p.289); traces (backward view,
Sec. 12.2 onward) remove the delay. Not needed by the page.

## 10. Outline feedback
- ch-ac: GAE (GAE paper Eq. 16, group E) has the same geometric weighting as Eq. 12.2, applied to TD residuals instead
  of returns. If the page shows the analogy, cite S&B Eq. 12.2 for the weights (1 - lambda) lambda^{n-1}, and Eq. 12.3
  for the lambda = 0 and lambda = 1 special cases.
- Name the range: Eq. 12.2 is printed with lambda in [0, 1); lambda = 1 is reached through Eq. 12.3. viz-gae can let
  the slider reach 1, but cite Eq. 12.3 for that end point.

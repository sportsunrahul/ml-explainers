---
chapter: ch-collapse
title: Why doesn't it collapse?
target_seconds: 160
---
Here is a solution to every prediction loss on this page. [[cue:btn-collapse-reset]]
Output the same vector for every input.
The target is constant, the prediction is constant, the distance is zero, and nothing has been learned.
I-JEPA's loss has no term against it.
So why does it learn anything at all?

Let me show you with a toy, built on a 2024 theory paper.
A linear encoder of depth two, a linear predictor, gradient descent on the squared error, and the target branch blocked from the gradient.
The data has a slow axis, where the two halves of a pair are nearly identical, and a noisy axis, where they are unrelated.
Press train. [[cue:btn-collapse-run]]
The cloud keeps its width along the slow axis, flattens along the noisy one, and the readouts settle near what the theory predicts.

Now guess, then switch the stop-gradient off and train again. [[cue:btn-collapse-sg-off]] [[cue:btn-collapse-run]]
Both branches are trained to agree, the cheapest agreement is a constant, and the cloud shrinks to a point.
Put it back and remove the predictor instead. [[cue:btn-collapse-sg-on]] [[cue:btn-collapse-pred-off]] [[cue:btn-collapse-run]]
A point again.
Those are the two asymmetries BYOL and SimSiam found by experiment in 2020, and I-JEPA adds a third: a slow average of the encoder as the target. [[cue:btn-collapse-pred-on]] [[cue:btn-collapse-ema-on]] [[cue:btn-collapse-run]]
In the toy, it behaves like the stop-gradient.

Why is the asymmetry enough?
For deep linear networks the paper works it out exactly.
Each feature of the data evolves on its own, with two numbers in charge.
Its covariance with the target sets how fast it is learned.
Its regression coefficient, how predictive it is per unit of variance, sets where its weight ends up: that coefficient raised to the depth. [[cue:eqstep-eq-fixedpoint-1]]
A clean predictable feature has a coefficient near one and is kept at full weight.
A noisy feature has a small one, and raised to the power of the depth, it nearly vanishes.
Pixel reconstruction gets a gentler exponent, always below one. [[cue:eqstep-eq-fixedpoint-2]]
So it keeps noisy features, however deep the encoder.
Switch the toy to depth one and back to see the noisy axis shrink with depth. [[cue:btn-collapse-ema-off]] [[cue:btn-collapse-l1]] [[cue:btn-collapse-run]]

There is a trap inside this gift.
A JEPA keeps whatever is slowest, and a fixed background is the perfect slow feature.
A 2022 paper showed it: a dot moving over noise that changes every frame is tracked fine; over noise that stays fixed, the encoder keeps the noise and loses the dot. [[cue:btn-collapse-l2]] [[cue:btn-collapse-bg]] [[cue:btn-collapse-run]]
The toy has that preset: the constant axis gets the largest weight of all.
V-JEPA's answer is to mask through time, so that copying the background does not solve the task.

And the other route has its own cost.
The methods that stop collapse with an explicit spread-out term are secretly doing k-means with equal clusters, so they assume balanced data.
Train them on batches of two classes and ImageNet accuracy drops by up to seventeen points; data2vec and MAE, which have no such term, barely move.
That is the fork this page keeps returning to, and LeJEPA, later, tries to get a loss term whose assumption is the right one.

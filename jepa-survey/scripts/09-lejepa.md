---
chapter: ch-lejepa
title: LeJEPA, one regulariser
target_seconds: 180
---
Everything so far avoided collapse with an asymmetry. [[cue:btn-sigreg-reset]]
A slow copy of the encoder, a stop-gradient, a predictor on one side only.
It works, but nobody can say what embedding distribution it is aiming at, and every new setup needs its schedules re-tuned.
LeJEPA starts from the other end.
First ask what the embeddings should look like for any downstream task.
Then write a single term that pushes them there, and drop the rest.

The argument runs through the probe.
After pretraining, someone will fit a linear classifier, or a nearest-neighbour rule, on the frozen embeddings, for a task nobody knows yet.
Which embedding distribution makes that probe's error smallest in the worst case, holding the total variance fixed?
Two lemmas show that stretching some directions and squashing others can only hurt a linear probe.
And a theorem shows that among all distributions with that variance, the isotropic Gaussian is the unique minimiser of the bias for nearest-neighbour and kernel probes.

So the target is known, and testing it in a thousand dimensions sounds hopeless.
But a classical trick makes it cheap.
A distribution is Gaussian if and only if its projection onto every direction is Gaussian.
So sample random directions, project the batch onto each, and run a one-dimensional test.
Here is the stretched cloud.
Guess which direction fails, then test along x. [[cue:btn-sigreg-dir-x]]
The projection is far too wide; the statistic is large.
Along z, the squashed axis, it fails the other way. [[cue:btn-sigreg-dir-z]]
A random direction mixes both. [[cue:btn-sigreg-dir-rand]]
Now press apply. [[cue:btn-sigreg-iso]]
The cloud eases to an isotropic Gaussian, and the statistic drops to its floor in every direction.
A clumpy cloud, by the way, fails even though its overall variance looks fine. [[cue:btn-sigreg-clumpy]] [[cue:btn-sigreg-iso]]

The test is Epps-Pulley: compare the empirical characteristic function of the projections with the Gaussian's. [[cue:eqstep-eq-sigreg-2]]
Average it over the sampled directions and you have SIGReg. [[cue:eqstep-eq-sigreg-1]]
Its gradient is bounded for any input, so it cannot blow up training.
And the whole method is two terms. [[cue:eqstep-eq-lejepa-1]]
Lambda times SIGReg on each view's batch, [[cue:eqstep-eq-lejepa-2]]
plus one minus lambda times a prediction term: every view's embedding pushed toward the mean of the global views.
One encoder, one hyperparameter, about fifty lines of code.
No target encoder, no moving average, no stop-gradient, no predictor, no prototypes.

The headline is stability rather than a record.
Seventy-nine percent linear accuracy on ImageNet for a ViT-H, the same recipe across more than sixty architectures up to nearly two billion parameters, a smooth loss curve, and a training loss that predicts downstream accuracy, so you can pick models without labels.
And on small in-domain datasets, pretraining from scratch beats transferring a frontier model: on eleven thousand galaxy images, a LeJEPA ResNet beats frozen DINOv3 by seven points.

Two companions close the chapter.
If the embeddings are Gaussian, the encoder's Jacobian tells you the data density, so a trained JEPA is a density estimator for free.
And under stationary latent dynamics, LeJEPA's two terms recover the world's true latents up to a linear map, and the Gaussian is the only prior for which that holds.

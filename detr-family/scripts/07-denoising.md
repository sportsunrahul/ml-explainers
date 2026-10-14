---
chapter: ch-denoising
title: DN-DETR and DINO: teach with noisy answers
target_seconds: 110
---
There's still a quieter problem in training. [[cue:btn-noise-reset]]
Early on, the matching is unstable.
A slot that's matched to the dog this epoch might be matched to the car the next.
Its target keeps flipping, so it keeps unlearning.

DN-DETR's fix is a piece of teaching folklore.
Give the student some problems where you already know the answer.
Take the real box, shift it and resize it a little, and feed these noisy copies to the decoder as extra queries. [[cue:btn-noise-dn]]
Each one has a fixed job: move back onto the true box.
No matching, no flipping. [[cue:btn-noise-sample]]

There's one catch.
The normal queries must not peek at these noisy copies, or they'd just copy near-perfect answers.
So an attention mask blocks that path.
And the mask really matters.
Without it, accuracy collapses to 24 AP, against 43.4 with it.

DINO pushes the idea further.
Plain denoising only teaches the model to fix a nearby box.
DINO adds hard negatives: boxes a bit further out that must be answered with "no object". [[cue:btn-noise-cdn]]
Green ones get fixed, red ones get rejected.
That teaches the model to turn down a near miss, which cuts duplicates.

DINO also starts its boxes from the encoder's best guesses for this image, and lets each layer learn from how much it helped the next one.
In the paper's ablation, those three ideas take a strong baseline from 46.5 to 47.9 AP.
And the four-scale DINO reaches 49 AP in only 12 epochs.
Remember where we started: 500 epochs for 42.

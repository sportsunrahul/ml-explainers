---
chapter: ch-ijepa
title: I-JEPA, predicting the missing block
target_seconds: 170
---
In January 2023 the diagram became a model. [[cue:btn-ijmask-reset]]
I-JEPA takes one image, hides a few large blocks, encodes the rest, and predicts the hidden blocks' embeddings.
No pixels are painted, no second augmented view is made.
And the two choices that make it work are both about the mask.

Look at the grid. [[cue:btn-ijmask-sample]]
Four yellow blocks are the targets, big enough that each one means something.
One blue square is the context, with every target patch removed from it.
On average a quarter of the image is left to look at.

Now follow the four parts.
The target encoder sees the whole image and produces one vector per patch; the yellow blocks select which vectors are the targets.
That order matters: the target for a leaf patch already knows it sits in a tree.
The context encoder sees only the blue patches.
And a narrow predictor takes the context vectors plus one mask token per target patch, a shared vector with that patch's position added, so it knows where it is being asked to guess.

The loss is simple. [[cue:eqstep-eq-ijepa-1]]
Average over the four blocks. [[cue:eqstep-eq-ijepa-2]]
Inside each block, sum over its patches the squared distance between the guess and the target vector. [[cue:eqstep-eq-ijepa-3]]
Nothing in it prevents collapse.
The target encoder is a slow moving average of the context encoder, and the predictor sits between them; why that is enough is the next chapter.

Here is a prediction for you.
What happens if the targets are tiny, single patches?
Press tiny targets. [[cue:btn-ijmask-tiny]]
Each hole can be filled by copying its neighbours, so nothing semantic needs to be learned, and the paper's accuracy falls from fifty-four percent to nineteen.
The other alternatives fare worse still. [[cue:btn-ijmask-raster]]
Predicting three quadrants from one gives fifteen and a half percent. [[cue:btn-ijmask-random]]
Random patches, seventeen point six. [[cue:btn-ijmask-multi]]
Multi-block, fifty-four.
And the question this page keeps asking has its cleanest answer here: the same model trained to predict pixels instead of embeddings scores forty point seven against sixty-six point nine, even though the pixel run trained longer.

What did it buy?
A frozen ViT-H reaches seventy-nine point three percent on ImageNet with a linear classifier, and a ViT-H at four forty-eight, eighty-one point one.
The augmentation-based methods are still a little ahead on that one number, but they lose on counting and depth, the low-level tasks their augmentations throw away.
And it trained on sixteen GPUs in under seventy-two hours, which the authors put at ten times less compute than MAE's model of the same size.

Now walk the stepper, and before stage seven, guess which of the three networks the gradient reaches. [[cue:btn-ijflow-reset]]
An image. [[cue:btn-ijflow-image]]
Patches with positions. [[cue:btn-ijflow-patches]]
The masks. [[cue:btn-ijflow-masks]]
Two encoders with one weight history. [[cue:btn-ijflow-encoders]]
The predictor fills the holes. [[cue:btn-ijflow-predictor]]
One number. [[cue:btn-ijflow-loss]]
And backward: the gradient reaches the predictor and the context encoder, and stops dead at the target encoder, which is brown. [[cue:btn-ijflow-backward]]
Then three updates: two optimizer steps, and the target encoder's slow step toward the new context encoder. [[cue:btn-ijflow-update]]

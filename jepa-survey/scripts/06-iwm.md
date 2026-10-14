---
chapter: ch-iwm
title: The predictor is a world model
target_seconds: 120
---
So far the predictor has been told one thing about the hole it fills: where it is. [[cue:btn-iwm-reset]]
Image World Models ask what happens if you also tell it what changed.
Build the target from one view of an image and the context from another, with a different color jitter, maybe blurred or turned grey.
Then hand the predictor the parameters of that difference.
Now it is not filling a hole; it is simulating a transformation in embedding space.
And that is what a world model does: given a state and an action, predict the next state.

Watch the toy.
The blue point is the context embedding; the three yellow points are the targets for three actions.
With the conditioned predictor, color jitter lands here, [[cue:btn-iwm-jitter]]
blur lands here, [[cue:btn-iwm-blur]]
grayscale here. [[cue:btn-iwm-gray]]
Now guess what must happen to the three targets if the predictor is blind to the action.
Press blind. [[cue:btn-iwm-blind]]
It has one output for all three, so the loss can only be small if the encoder puts the three targets in one place.
Color, blur and grayscale get discarded.
That is invariance, and it is exactly what I-JEPA with jittered views would learn.
Tell the predictor, give it enough capacity, and the encoder may keep color while the predictor learns to shift it. [[cue:btn-iwm-cond]]
The paper calls that an equivariant world model.

The loss is I-JEPA's with one extra input: the action vector, the differences in brightness, contrast, saturation and hue, plus flags for the destructive augmentations. [[cue:eqstep-eq-iwm-1]]
It is still a squared distance to the target encoder's embedding. [[cue:eqstep-eq-iwm-2]]
And whether the world model actually becomes equivariant is measured directly: can the predictor rank the right transformed image first out of a bank of two hundred and fifty-six?
With a twelve-layer predictor, almost never; with eighteen layers and strong jitter, four times out of five.
Conditioning, capacity and a hard enough transformation all have to be there.

Why care?
Because once the predictor is a capable world model, you can fine-tune the predictor on a downstream task, with the encoder frozen, and match fine-tuning the whole encoder.
Hold on to the shape of that result.
An action-conditioned predictor over frozen features is exactly what V-JEPA 2 does with robot actions, two chapters from now.

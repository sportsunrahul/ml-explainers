---
chapter: ch-set-problem
title: Detection is a set problem
target_seconds: 100
---
Look at a photo and tell me what's in it. [[cue:btn-nms-reset]]
You'd say something like: two people, a dog, a car.
No order, and no repeats.
That's a set, and it's exactly what an object detector is supposed to give us.

Here's the funny thing.
Classic detectors never produce that set directly.
They throw down thousands of candidate boxes, score every one, and end up with a little cloud of guesses around each object.
That's what you're looking at: two people, and a swarm of blue boxes.

So how do you get down to one box per person?
With a cleanup step called non-maximum suppression. [[cue:btn-nms-run]]
Keep the most confident box, and delete anything that overlaps it too much.
With the people standing apart, that works nicely.

Now push them together. [[cue:btn-nms-close]]
Make the rule strict, and the second person simply vanishes. [[cue:btn-nms-low]]
NMS can't tell another person from the same person again.
Loosen it, and now the duplicates pile up instead. [[cue:btn-nms-high]]
Try the slider yourself, and ask whether one setting could be right for every scene.

Here's the real point.
The network never learned to avoid duplicates.
A hand-tuned rule cleans up after it.
So DETR asks a simple question: can a network just output the final set, directly, with nothing to clean up?

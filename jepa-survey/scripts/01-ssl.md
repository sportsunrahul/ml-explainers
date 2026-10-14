---
chapter: ch-ssl
title: Three ways to fill in a blank
target_seconds: 150
---
Cover part of a photo with your hand. [[cue:btn-blank-reset]]
You can still say what is under it: more sky, more leaves, the rest of the road.
Nobody labelled that photo for you.
That is the whole idea of self-supervised learning: hide part of the data, make the network guess, and grade the guess.
And the question that runs through this entire page is where you grade it.

Here is the first option. [[cue:btn-blank-pixels]]
Guess the hidden pixels, and grade every single one.
Masked autoencoders do this at scale: hide three quarters of the image, paint the rest back in.
Look at the guess, though.
For a patch of leaves, the best guess is a blur, and the loss punishes every leaf the blur got wrong.
No one could have known where that leaf was, but the network pays for it anyway.

Now press the noise button. [[cue:btn-blank-noisy]]
I have added noise to the hidden pixels, and the pixel loss jumps.
Hold on to that: a pixel loss has to spend capacity on things that cannot be predicted.

Second option. [[cue:btn-blank-views]]
Refuse to predict anything.
Take two crops of the same image and push their embeddings together.
This is the family of SimCLR, BYOL, DINO and VICReg.
The catch is that the easiest way to make two things agree is to make them both the same constant.
That failure has a name, collapse, and it gets its own chapter.
The other catch is subtler: you have told the model that the two crops are the same, so whatever differs between them is thrown away on purpose.

Third option, and the one this page is about. [[cue:btn-blank-embed]]
Hide part of the input, but predict its embedding, not its pixels.
A short vector that says what is there, graded against the encoder's own description of the hidden part.
In the toy, four numbers are graded instead of nearly two hundred pixels.
Press noise again and watch: the embedding loss barely moves. [[cue:btn-blank-noisy]]
A real encoder is free to learn a description that keeps "leaves, this green, this texture" and drops "this exact leaf here".
That freedom is the whole idea.
The rest of the page is how to get it without collapsing.

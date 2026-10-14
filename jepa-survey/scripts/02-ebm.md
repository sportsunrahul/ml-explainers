---
chapter: ch-ebm
title: The energy view
target_seconds: 150
---
Before any of this was a model, it was a diagram. [[cue:btn-energy-reset]]
In 2022 Yann LeCun posted a position paper, and the recipe this whole page follows is drawn in it.
It rests on one change of viewpoint.
Stop thinking of a model as a function that outputs an answer.
Think of it as a function that scores how well an answer fits.

That score is called an energy. [[cue:eqstep-eq-energy-1]]
Low energy means these two things go together; high energy means they do not.
Look at the surface: two valleys, where the data lives.
To answer a question, you do not run the network once.
You search for the answer with the lowest energy, by sliding downhill. [[cue:eqstep-eq-energy-2]]

Why go to the trouble?
Because a function can give only one answer, and the hidden block could contain a hundred arrangements of leaves.
A surface can have many valleys.
And any leftover ambiguity goes into a small extra input, a latent variable. [[cue:eqstep-eq-latent-2]]
The energy of a pair is the best energy over all choices of that latent.

Now here is the trap, and I want you to guess first.
Suppose the energy is just the distance between two embeddings, one for the input and one for the answer.
What does the landscape look like if both encoders output the same constant for everything?
Press collapsed. [[cue:btn-energy-flat]]
Flat.
Every answer is equally good, and the model has learned nothing.
The paper calls this collapse, and it is the price of predicting in embedding space: someone has to pay for the shape of the landscape.

Two families pay.
Contrastive methods pick wrong answers and push their energy up. [[cue:btn-energy-contrastive]]
It works where you push; in a high-dimensional space there are too many places to push.
Regularised methods instead limit how much low-energy space there can be at all. [[cue:btn-energy-reg]]
Hold on to this split, because every anti-collapse trick on this page is one of the two.

So here is the diagram. [[cue:btn-energy-trained]]
Two encoders turn the input and the answer into embeddings. [[cue:eqstep-eq-jepa-1]]
A predictor, given the input's embedding and the latent, guesses the answer's embedding. [[cue:eqstep-eq-jepa-3]]
The energy is the distance between the guess and the real thing. [[cue:eqstep-eq-jepa-2]]
That is a joint-embedding predictive architecture.
And three things follow from the picture.
The answer encoder may discard what cannot be predicted, which is a feature for noise and a bug if it discards everything.
If the input is the past and the answer is the future, the predictor is a world model.
And you can stack them, a second one predicting further ahead from coarser embeddings.
That last box waited four years for a working version.

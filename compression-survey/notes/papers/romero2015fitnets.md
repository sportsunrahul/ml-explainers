# romero2015fitnets -- FitNets: Hints for Thin Deep Nets (BRANCH)

## Identity
- arXiv 1412.6550, version read: v4 (2015-03-27). v1 submitted 2014-12-19 (abs.html).
- Authors: Adriana Romero ... Yoshua Bengio (6 authors). Venue: ICLR 2015 (TeX file `iclr2015.tex`, ICLR
  style; abs.html has no journal ref).
- Repo (paper footnote, Sec. 3): https://github.com/adri-romsor/FitNets (not downloaded).
- Custom macro in the quoted TeX: `\argminl` = `\argmin\limits`.

## Key idea (two sentences)
A thin, deep student ("FitNet") is first trained so that a middle layer ("guided" layer), passed through a small
convolutional regressor, predicts a middle layer ("hint" layer) of a trained, wider teacher. Then the whole
student is trained with Hinton-style KD (Eq. 2), starting from that pre-trained initialization.

## Equations (TeX verbatim, `src/iclr2015.tex`; numbers from PDF p. 3)
Eq. 1, softened outputs:
```
\mathrm{P_T^\tau} = \mathrm{softmax}\left(\frac{\mathbf{a}_T}{\tau}\right),\quad \mathrm{P_S^\tau} = \mathrm{softmax}\left(\frac{\mathbf{a}_S}{\tau}\right).
```
Eq. 2 (`eq:toplayer`), KD loss (no tau^2 factor; lambda weights the soft term):
```
\mathcal{L}_{KD}(\mathbf{W_S}) = \mathcal{H}(\mathbf{y_{true}},\mathrm{P_S}) +
\lambda \mathcal{H}(\mathrm{P_T^\tau}, \mathrm{P_S^\tau}),
```
Eq. 3 (`eq:hints`), hint loss:
```
\mathcal{L}_{HT}(\mathbf{W_{Guided}},\mathbf{W_r}) = \frac{1}{2} || u_{h}(\mathbf{x};\mathbf{W_{Hint}}) - r(v_{g}(\mathbf{x};\mathbf{W_{Guided}});\mathbf{W_r}) ||^2,
```
Paper: "$u_{h}$ and $v_{g}$ are the teacher/student deep nested functions up to their respective hint/guided
layers with parameters $\mathbf{W_{Hint}}$ and $\mathbf{W_{Guided}}$, $r$ is the regressor function on top of
the guided layer with parameters $\mathbf{W_r}$."
Gloss: run the image through the teacher up to the hint layer (fixed target) and through the student up to the
guided layer; the regressor maps student features to teacher shape; half squared L2 distance. Only
W_Guided and W_r are arguments of the loss (teacher W_Hint is given) -> teacher fixed (`our reading` of the
argument list and Algorithm 1, whose input is "the trained parameters $\mathbf{W_T}$ of a teacher").
Regressor: convolutional, kernel k1 x k2 chosen so that N_{g,i} - k_i + 1 = N_{h,i}; params
k1 x k2 x O_h x O_g, instead of a fully connected N_h1 x N_h2 x O_h x N_g1 x N_g2 x O_g (Sec. 2.2).

## Two-stage procedure (Sec. 2.3, Algorithm 1, p. 4)
1. Stage 1: from a trained teacher and a random FitNet, add regressor W_r; train W_Guided (student layers 1..g)
   and W_r to minimize Eq. 3.
2. Stage 2: copy W_Guided* into the student; drop the regressor; train all student weights W_S to minimize
   Eq. 2. Lambda annealed linearly "to promote the learning of more complex examples" (Sec. 2.4).

## Config (Appendix A.1, p. 11; Sec. 3.1)
- CIFAR-10 teacher: maxout conv net of Goodfellow 2013, 3 conv layers 96-192-192 + maxout FC 500.
- FitNet (Tables 1-2): 19 layers (FitNet 4), 3x3 conv + maxout; "The 11th layer of the student network was
  trained to mimic the 2nd layer of the teacher network."
- RMSProp, lr 0.005, mini-batch 128; lambda initialized to 4, decayed linearly over 500 epochs to 1; tau = 3.

## Key numbers
- Table 1 (p. 5, CIFAR-10 accuracy): FitNet ~2.5M params 91.61%; Teacher ~9M 90.18%. Best mimic (Ba & Caruana)
  "Mimic ensemble" ~70M 85.8%.
- Table 5 (p. 8, CIFAR-10 accuracy/speed trade-off), columns Network | # layers | # params | # mult | Acc |
  Speed-up | Compression rate:
| Network | layers | params | mult | Acc | Speed-up | Compression |
|---|---|---|---|---|---|---|
| Teacher | 5 | ~9M | ~725M | 90.18% | 1 | 1 |
| FitNet 1 | 11 | ~250K | ~30M | 89.01% | 13.36 | 36 |
| FitNet 2 | 11 | ~862K | ~108M | 91.06% | 4.64 | 10.44 |
| FitNet 3 | 13 | ~1.6M | ~392M | 91.10% | 1.37 | 5.62 |
| FitNet 4 | 19 | ~2.5M | ~382M | 91.61% | 1.52 | 3.60 |
- Speed: "total inference times required for processing CIFAR-10 test examples on a GPU" (Sec. 4.2). GPU model
  and batch size not stated.

## Outline feedback
- FitNets KD (Eq. 2) has no tau^2 factor and puts lambda on the soft term; do not cite it for the T^2 form.
- Wang 2019 Table 1 row 0.25x-F shows FitNets-style full feature imitation hurts a detector (-8.9 mAP). That
  pairs well with the "logits are not enough, but whole-feature hints are too much" line.

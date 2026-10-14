# BYOL (branch)

- Title: Bootstrap your own latent: A new approach to self-supervised Learning. arXiv 2006.07733, read v3 (2020-09-10); v1 2020-06-13.
- Authors: Jean-Bastien Grill ... Michal Valko. Venue not stated in abs.html (src has a commented `neurips_2020` preprint option).
- Key idea: an online network (encoder, projector, predictor) predicts the target network's projection of another view; the target is an exponential moving average of the online weights and gets no gradient. No negatives.
- Defining equation (`eq:cosine-loss`, src/method.tex; `\netparams`=theta, `\targetparams`=xi, `\eqdef`=triangleq):
```
\mathcal{L}_{\netparams, \targetparams} \eqdef \normtwo{\bar{q_{\theta}}(z_\netparams) - \bar{z}'_\targetparams}^2 = 2 - 2\cdot\frac{\langle q_\netparams(z_\netparams),  z'_\targetparams \rangle }{\big\|q_\netparams(z_\netparams)\big\|_2\cdot \big\|z'_\targetparams\big\|_2 }
```
  Symmetrized; minimized w.r.t. theta only. EMA `\targetparams \leftarrow \tau\targetparams + (1 - \tau) \netparams`, tau_base = 0.996 increased to 1 (Sec 3.3).
- Config (Sec 3.3): ResNet-50 (2048) -> MLP 4096 -> 256; predictor same shape; LARS, 1000 epochs, batch 4096 on 512 Cloud TPU v3 cores, about 8 hours.
- Key number: 74.3 ImageNet linear top-1 (top-5 91.6), ResNet-50 1x, 1000 ep, batch 4096, two 224 views, no multi-crop (Table 1a, row BYOL (ours)).
- Table 1a also prints: SimCLR 69.3, MoCo v2 71.1, InfoMin Aug. 73.0, CMC 66.2, CPC v2 63.8, PIRL 63.6, Local Agg. 60.2.
- Outline feedback: one paragraph + link to JEPA survey is right. BYOL is the clearest example of "no negatives, asymmetry prevents collapse".

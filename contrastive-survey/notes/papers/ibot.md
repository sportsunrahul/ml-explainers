# iBOT (branch)

- Title: iBOT: Image BERT Pre-Training with Online Tokenizer. arXiv 2111.07832, read v3 (2022-01-27); v1 2021-11-15.
- Authors: Jinghao Zhou ... Tao Kong. Venue: ICLR 2022 (src file iclr2022_conference.tex; not in abs.html comment).
- Key idea: DINO's [CLS] self-distillation plus masked image modeling where the EMA teacher is the "online tokenizer": the student predicts the teacher's patch-token distributions at masked positions.
- Defining equation (`eq:mim_obj`; `\vthetap` = teacher params):
```
\mathcal{L}_{\mathrm{MIM}} = - \sum_{i=1}^N \ m_i \cdot P_\vthetap^{\mathrm{patch}}(\vu_i)^\mathrm{T} \ \mathrm{log} \ P_\vtheta^{\mathrm{patch}}(\hat \vu_i).
```
  m_i = 1 at masked patches; u-hat = masked view seen by the student. Plus `\mathcal{L}_{\texttt{[CLS]}}` (`eq:cls_obj`).
- Key number: ViT-L/16 81.0 linear, 78.0 k-NN, ImageNet-1K pretraining, 1200 effective epochs (Table 1). Abstract headline 82.3 linear is ViT-L/16 pretrained on ImageNet-22K (Table 1, row marked double dagger, 200 epochs).
- ViT-B/16: 79.5 linear, 77.1 k-NN, 1600 effective epochs (Table 1). Fine-tune headline 87.8 (ViT-L/16, 512 res, IN-22K).
- "Epo." in Table 1 is effective epochs counting all views (footnote).

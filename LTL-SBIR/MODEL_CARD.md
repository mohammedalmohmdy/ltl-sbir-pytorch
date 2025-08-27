# LTL-SBIR — Model Card

**Date:** 2025-08-27
**Version:** 1.0.0
**Repository:** (to be added after GitHub push)

## Overview
LTL-SBIR is a PyTorch implementation of a Sketch-Based Image Retrieval system that learns frequency-aware representations via a Learnable Transform Layer (LTL) placed after a CNN backbone and before the embedding head. It targets fine-grained instance-level retrieval between hand-drawn sketches (queries) and natural photos (gallery).

## Intended Use
- Primary: Academic research and benchmarking on public SBIR datasets (QMUL-Shoe-V2, QMUL-Chair, Sketchy (Extended)).
- Out-of-scope / Not intended for: Biometric identification, surveillance, or any use with personal/sensitive images without consent and compliance with applicable laws.

## Model Architecture
- Backbones: ResNet-18/50, VGG16 (torchvision).
- Learnable Transform Layer (LTL): Depthwise 3×3 -> Pointwise 1×1 (+BN/ReLU) -> optional SE (Squeeze-and-Excitation) -> 1×1 projection + residual.
- Heads: GlobalAvgPool -> Linear -> BN -> L2 normalization (embedding).
- Training: Batch-Hard Triplet Loss; AMP mixed precision; optional explicit triplets or batch-hard mining.

## Datasets
The repository expects CSV metadata and paths to images/sketches. Users must supply and agree to terms of use of the datasets. Example layouts are included under examples/.

## Training & Evaluation
- Training: `python src/train.py --config configs/default.yaml`
- Evaluation: `python src/evaluate.py --config configs/default.yaml --ckpt runs/ltl_sbir/best.ckpt` (outputs `eval.json` with mAP, P@K, CMC)
- Plotting: `python src/scripts/plot_curves.py --eval_json runs/ltl_sbir/eval.json --out_prefix runs/ltl_sbir/plots`

## Metrics
- mAP (Mean Average Precision)
- P@K (Precision at K; K in {1,5,10,100} configurable)
- CMC (Cumulative Match Characteristic)

Reproducibility: Set seeds in `configs/default.yaml`. Minor variations may occur due to nondeterminism in CUDA/cuDNN.

## Risks & Limitations
- Domain shift: Trained on object-centric datasets; may not generalize to scene-level sketches/photos.
- Style sensitivity: Extreme sketch abstraction, stylization, or adversarial perturbations can degrade retrieval.
- Compute: Though lightweight vs. GANs, inference still depends on the backbone size and gallery indexing strategy.
- Ethical: Avoid using on datasets with privacy concerns; ensure annotations and usage comply with licenses, privacy, and consent requirements.

## Ethical Considerations
- Use only with datasets/images where you have rights and consent.
- Do not deploy in contexts where retrieval could cause harm, bias, or unfair profiling.
- Provide transparency when used in demos or publications; highlight dataset sources and limitations.

## How to Cite
If you use this implementation, please cite the original paper and this repository:
```
@software{LTL_SBIR_Repo_2025,
  title        = {LTL-SBIR: Learnable Transform Layer for Sketch-Based Image Retrieval (PyTorch)},
  author       = {Al-Mohamadi, Mohammed A.S and Prabhakar, C J},
  year         = {2025},
  url          = {https://github.com/mohammedalmohmdy/ltl-sbir-pytorch}
}
```
And cite the academic paper as specified in your manuscript.

## Repro Tips
- Start with ResNet-50, embedding dim 512, margin 0.2.
- Batch size 32 (increase if memory allows); enable AMP.
- Ensure labels in CSV are instance IDs, not categories.
- For faster retrieval, precompute gallery embeddings (see `src/scripts/precompute_embeddings.py`).

## Contact & Support
Please open a GitHub Issue with minimal reproduction and system details.


# LTL‑SBIR (Learnable Transform Layer for Sketch‑Based Image Retrieval)

Complete PyTorch implementation of **LTL‑SBIR**: a Siamese/Triplet network with a **Learnable Transform Layer** (depthwise + pointwise + SE) for frequency‑aware cross‑domain matching between sketches and photos.

> Datasets not included. Point the config to local copies of **QMUL‑Shoe‑V2**, **QMUL‑Chair**, or **Sketchy (Extended)**.

## Install
```bash
conda create -n ltlsbir python=3.10 -y && conda activate ltlsbir
pip install -r requirements.txt
```

## Data Layout (example)
```
data/
  qmul-shoe-v2/
    images/
    sketches/
    meta/
      images.csv    # columns: path,label
      sketches.csv  # columns: path,label
      train_pairs.csv   # OPTIONAL: sketch_path,pos_img_path,neg_img_path
```
Update paths in `configs/default.yaml`.

## Train / Eval / Retrieve
```bash
python src/train.py --config configs/default.yaml
python src/evaluate.py --config configs/default.yaml --ckpt runs/ltl_sbir/best.ckpt
python src/scripts/precompute_embeddings.py --config configs/default.yaml --ckpt runs/ltl_sbir/best.ckpt
python src/retrieve.py --config configs/default.yaml --ckpt runs/ltl_sbir/best.ckpt --query path/to/sketch.png --topk 10
```

Artifacts: checkpoints & logs in `runs/ltl_sbir/`.


---


```bash
git init
git add .
git commit -m "Initial LTL-SBIR release"
git branch -M main
git remote add origin https://github.com/mohammedalmohmdy/ltl-sbir-pytorch.git
git push -u origin main
```

```bash
python src/evaluate.py --config configs/default.yaml --ckpt runs/ltl_sbir/best.ckpt
python src/scripts/plot_curves.py --eval_json runs/ltl_sbir/eval.json --out_prefix runs/ltl_sbir/plots
```



See [MODEL_CARD.md](MODEL_CARD.md) for details on architecture, training, and limitations.

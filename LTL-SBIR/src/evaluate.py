
import os, argparse, yaml, json
import torch, numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.sketch_photo import build_eval_sets
from models.ltl_sbir import SiameseLTLNet
from utils.metrics import mean_average_precision, cmc_curve
from utils.common import ensure_dir

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    return ap.parse_args()

def embed_all(model, items, tf, device, branch="image", bs: int = 64):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(items), bs), desc=f"Embed {branch}"):
            batch = items[i:i+bs]
            tensor = torch.stack([tf(Image.open(p).convert("RGB")) for p,_ in batch]).to(device)
            e = model.forward_branch(tensor, branch=branch)
            embs.append(e.cpu())
            labels.extend([lbl for _,lbl in batch])
    return torch.cat(embs, dim=0).numpy(), np.array(labels)

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["experiment"]["out_dir"]; ensure_dir(out_dir)

    model = SiameseLTLNet(
        backbone=cfg["model"]["backbone"],
        emb_dim=int(cfg["model"]["embedding_dim"]),
        shared_backbone=bool(cfg["model"]["shared_backbone"]),
        ltl_pw_channels=int(cfg["model"]["ltl"]["pw_channels"]),
        ltl_dw_kernel=int(cfg["model"]["ltl"]["dw_kernel"]),
        ltl_use_se=bool(cfg["model"]["ltl"]["use_se"]),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    sketches, images, tf = build_eval_sets(cfg["data"]["root"], cfg["data"]["sketches_csv"], cfg["data"]["images_csv"], int(cfg["data"]["img_size"]))
    img_embs, img_labels = embed_all(model, images, tf, device, branch="image")
    sk_embs, sk_labels = embed_all(model, sketches, tf, device, branch="sketch")

    a2 = np.sum(sk_embs**2, axis=1, keepdims=True)
    b2 = np.sum(img_embs**2, axis=1, keepdims=True).T
    cross = sk_embs @ img_embs.T
    dists = np.sqrt(np.maximum(a2 + b2 - 2*cross, 0.0))
    ranks = np.argsort(dists, axis=1)

    Kmax = max(cfg["eval"]["topk"])
    all_rel = []
    metrics = {}
    for qi in range(sk_embs.shape[0]):
        qlbl = sk_labels[qi]
        labels_sorted = img_labels[ranks[qi]]
        binary = [1 if lbl == qlbl else 0 for lbl in labels_sorted[:Kmax]]
        all_rel.append(binary)

    mAP = mean_average_precision(all_rel)
    metrics["mAP"] = float(mAP)
    for k in cfg["eval"]["topk"]:
        p_at_k = float(np.mean([np.mean(r[:k]) for r in all_rel]))
        metrics[f"P@{k}"] = p_at_k

    cmc = cmc_curve(all_rel, max_rank=Kmax)

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump({"metrics": metrics, "cmc@1..K": cmc[:10]}, f, indent=2)

    print("==== Evaluation ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("CMC@1..10:", cmc[:10])

if __name__ == "__main__":
    main()

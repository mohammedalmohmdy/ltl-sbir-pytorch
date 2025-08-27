
import os, argparse, yaml, json, numpy as np, torch
from PIL import Image
from tqdm import tqdm

from datasets.sketch_photo import build_eval_sets
from models.ltl_sbir import SiameseLTLNet
from utils.common import ensure_dir

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    return ap.parse_args()

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
    model.eval()

    sketches, images, tf = build_eval_sets(cfg["data"]["root"], cfg["data"]["sketches_csv"], cfg["data"]["images_csv"], int(cfg["data"]["img_size"]))

    # gallery embeddings
    g_paths, g_labels, g_embs = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), 64), desc="Embed gallery"):
            batch = images[i:i+64]
            tensor = torch.stack([tf(Image.open(p).convert("RGB")) for p,_ in batch]).to(device)
            e = model.forward_branch(tensor, branch="image").cpu().numpy()
            g_embs.append(e); g_paths.extend([p for p,_ in batch]); g_labels.extend([int(lbl) for _,lbl in batch])
    import numpy as np
    g_embs = np.vstack(g_embs)

    # query
    with torch.no_grad():
        q = tf(Image.open(args.query).convert("RGB")).unsqueeze(0).to(device)
        q_emb = model.forward_branch(q, branch="sketch").cpu().numpy()

    # distances
    a2 = np.sum(q_emb**2, axis=1, keepdims=True)
    b2 = np.sum(g_embs**2, axis=1, keepdims=True).T
    cross = q_emb @ g_embs.T
    d = np.sqrt(np.maximum(a2 + b2 - 2*cross, 0.0)).flatten()
    order = np.argsort(d)[:args.topk]

    results = [{"rank": int(i+1), "path": g_paths[idx], "label": int(g_labels[idx]), "dist": float(d[idx])} for i, idx in enumerate(order)]
    out_json = os.path.join(out_dir, "retrieval.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("Top‑K:")
    for r in results: print(r)
    print("Saved →", out_json)

if __name__ == "__main__":
    main()

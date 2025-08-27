
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
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    out_dir = cfg["experiment"]["out_dir"]; ensure_dir(out_dir)

    def embed(items, branch):
        paths, labels, embs = [], [], []
        with torch.no_grad():
            for i in tqdm(range(0, len(items), 64), desc=f"Embed {branch}"):
                batch = items[i:i+64]
                import numpy as np
                import torch
                from PIL import Image
                from torchvision import transforms
                tensor = torch.stack([tf(Image.open(p).convert("RGB")) for p,_ in batch]).to(device)
                e = model.forward_branch(tensor, branch=branch).cpu().numpy()
                embs.append(e); paths.extend([p for p,_ in batch]); labels.extend([int(lbl) for _,lbl in batch])
        return np.vstack(embs), paths, labels

    g_embs, g_paths, g_labels = embed(images, "image")
    q_embs, q_paths, q_labels = embed(sketches, "sketch")

    np.save(os.path.join(out_dir, "gallery_embs.npy"), g_embs)
    np.save(os.path.join(out_dir, "query_embs.npy"), q_embs)
    json.dump({"paths": g_paths, "labels": g_labels}, open(os.path.join(out_dir, "gallery_meta.json"), "w"))
    json.dump({"paths": q_paths, "labels": q_labels}, open(os.path.join(out_dir, "query_meta.json"), "w"))
    print("Saved precomputed embeddings to", out_dir)

if __name__ == "__main__":
    main()


import os, argparse, yaml, time, csv
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from datasets.sketch_photo import SketchPhotoPairDataset
from models.ltl_sbir import SiameseLTLNet
from losses import BatchHardTripletLoss
from utils.common import set_seed, ensure_dir, count_parameters

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def make_loaders(cfg):
    ds = SketchPhotoPairDataset(
        root=cfg["data"]["root"],
        sketches_csv=cfg["data"]["sketches_csv"],
        images_csv=cfg["data"]["images_csv"],
        train_pairs_csv=cfg["data"]["train_pairs_csv"],
        img_size=int(cfg["data"]["img_size"]),
        split="train"
    )
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                    num_workers=int(cfg["data"]["num_workers"]), drop_last=True, pin_memory=True)
    return dl

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(int(cfg["experiment"]["seed"]))
    out_dir = cfg["experiment"]["out_dir"]
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseLTLNet(
        backbone=cfg["model"]["backbone"],
        emb_dim=int(cfg["model"]["embedding_dim"]),
        shared_backbone=bool(cfg["model"]["shared_backbone"]),
        ltl_pw_channels=int(cfg["model"]["ltl"]["pw_channels"]),
        ltl_dw_kernel=int(cfg["model"]["ltl"]["dw_kernel"]),
        ltl_use_se=bool(cfg["model"]["ltl"]["use_se"]),
    ).to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    if cfg["train"]["optimizer"].lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    else:
        optim = torch.optim.SGD(model.parameters(), lr=float(cfg["train"]["lr"]), momentum=0.9, weight_decay=float(cfg["train"]["weight_decay"]))

    criterion = BatchHardTripletLoss(margin=float(cfg["train"]["margin"]))
    scaler = GradScaler(enabled=bool(cfg["train"]["mixed_precision"]))
    dl = make_loaders(cfg)

    log_path = os.path.join(out_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss"])

    best_loss = 1e9
    ckpt_path = os.path.join(out_dir, "best.ckpt")

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        epoch_loss, n = 0.0, 0
        start = time.time()
        for batch in dl:
            optim.zero_grad(set_to_none=True)

            if cfg["data"]["train_pairs_csv"]:
                s = batch["sketch"].to(device, non_blocking=True)
                ip = batch["img_pos"].to(device, non_blocking=True)
                ineg = batch["img_neg"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True).long()
                e_s = model.forward_branch(s, branch="sketch")
                e_p = model.forward_branch(ip, branch="image")
                e_n = model.forward_branch(ineg, branch="image")
                emb = torch.cat([e_s, e_p, e_n], dim=0)
                lab = torch.cat([y, y, y+10_000], dim=0)
            else:
                s = batch["sketch"].to(device, non_blocking=True)
                i = batch["image"].to(device, non_blocking=True)
                ys = batch["y_sketch"].to(device, non_blocking=True).long()
                yi = batch["y_image"].to(device, non_blocking=True).long()
                e_s = model.forward_branch(s, branch="sketch")
                e_i = model.forward_branch(i, branch="image")
                emb = torch.cat([e_s, e_i], dim=0)
                lab = torch.cat([ys, yi], dim=0)

            with autocast(enabled=bool(cfg["train"]["mixed_precision"])):
                loss = criterion(emb, lab)

            scaler.scale(loss).backward()
            if float(cfg["train"]["grad_clip"]) > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optim); scaler.update()

            epoch_loss += loss.item(); n += 1

        epoch_loss /= max(n, 1)
        dur = time.time() - start
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']}  loss={epoch_loss:.4f}  time={dur:.1f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{epoch_loss:.6f}"])

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

    print(f"Done. Best loss={best_loss:.4f}. Saved: {ckpt_path}")

if __name__ == "__main__":
    main()

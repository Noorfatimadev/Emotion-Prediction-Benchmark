"""
Part 1: Bag-of-Words baseline experiments.

Grid search over:
- BoW variants: count, binary, log_tf, tfidf
- number of MLP layers

Reports metrics on train and dev:
MSE, MAE, RMSE, Pearson, Spearman

Saves:
- best checkpoint per (bow_variant, n_layers, seed) based on dev RMSE
- JSONL logs for summaries + per-epoch history
"""

import os
import json
import math
import random
import logging
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from datasets1 import BowDataset
from model import MLPModel


# Reproducibility

def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy and PyTorch (CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Evaluation

@torch.no_grad()
def evaluate(regressor: nn.Module, data_iter: DataLoader, device: torch.device) -> dict:
    """Evaluate model and return regression metrics."""
    regressor.eval()

    y_true_chunks, y_pred_chunks = [], []

    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)

        out = regressor(x).squeeze(-1)
        y_pred_chunks.append(out.detach().cpu().numpy())
        y_true_chunks.append(y.squeeze(-1).detach().cpu().numpy())

    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    # Avoid correlation issues with constant arrays
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        pear = float("nan")
        spear = float("nan")
    else:
        pear = float(pearsonr(y_true, y_pred)[0])
        spear = float(spearmanr(y_true, y_pred)[0])

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": pear,
        "spearman": spear,
    }


# Data split

def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    seed: int,
    n_bins: int = 10,
):
    """
    Approximate stratification for regression by binning labels into quantiles.
    Keeps train/dev label distributions similar.
    """
    y = df[label_col].values
    ranks = pd.Series(y).rank(method="average").values
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")

    train_df, dev_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=bins,
    )
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


#  Checkpoint saving

def save_model(path: str, args, model: nn.Module, vocab: dict, bow_idf):
    """Save model checkpoint + vocabulary (+ idf if tfidf)."""
    obj = {
        "args": args,
        "model": model.state_dict(),
        "vocab": vocab,
    }
    if bow_idf is not None:
        obj["bow_idf"] = bow_idf

    torch.save(obj, path)


#  Training

def train_one_config(
    args,
    train_df,
    dev_df,
    device,
    bow_variant,
    n_layers,
    run_seed,
    out_dir,
):
    """Train one configuration and return metrics + history."""
    cfg = deepcopy(args)
    cfg.bow = True
    cfg.bow_variant = bow_variant
    cfg.n_layers = n_layers
    cfg.seed = run_seed

    set_seed(run_seed)

    # Build training dataset (creates vocab / idf)
    train_ds = BowDataset(
        train_df,
        cfg.bow_vocab_size,
        cfg.data_col,
        split="train",
        lowercase=cfg.lowercase,
        vocab=None,
        bow_variant=cfg.bow_variant,
        idf=None,
    )
    vocab = train_ds.vocab
    bow_idf = train_ds.get_idf() if cfg.bow_variant == "tfidf" else None

    # Dev uses same vocab / idf
    dev_ds = BowDataset(
        dev_df,
        cfg.bow_vocab_size,
        cfg.data_col,
        split="dev",
        lowercase=cfg.lowercase,
        vocab=vocab,
        bow_variant=cfg.bow_variant,
        idf=bow_idf,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = MLPModel(cfg, input_dim=cfg.bow_vocab_size, output_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_dev_rmse = float("inf")
    best_state = None
    best_epoch = -1

    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        train_metrics = evaluate(model, train_loader, device)
        dev_metrics = evaluate(model, dev_loader, device)

        row = {
            "bow_variant": cfg.bow_variant,
            "n_layers": cfg.n_layers,
            "seed": cfg.seed,
            "epoch": epoch,
            "train_loss_mse": float(total_loss / max(1, len(train_loader))),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(row)

        # Track best dev RMSE
        if dev_metrics["rmse"] < best_dev_rmse:
            best_dev_rmse = dev_metrics["rmse"]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

    cfg_name = f"bow_{cfg.data_col}_{cfg.bow_variant}_layers{cfg.n_layers}_seed{cfg.seed}"
    ckpt_path = os.path.join(out_dir, f"{cfg_name}.bin")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_model(ckpt_path, cfg, model, vocab=vocab, bow_idf=bow_idf)

    return {
        "config_name": cfg_name,
        "checkpoint": ckpt_path,
        "best_epoch": best_epoch,
        "best_dev_rmse": best_dev_rmse,
        "history": history,
        "final_train_metrics": evaluate(model, train_loader, device),
        "final_dev_metrics": evaluate(model, dev_loader, device),
    }


# Main

def main():
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    log = logging.getLogger("bow_experiments")

    ap = ArgumentParser()
    ap.add_argument("--data_path", default="emo_dataset.tsv")
    ap.add_argument("--out_dir", default="runs_bow")
    ap.add_argument("--data_col", default="lemma", choices=["text", "lemma", "PoS"])
    ap.add_argument("--label_col", default="normalized")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--bow_vocab_size", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    # grid
    ap.add_argument("--bow_variants", nargs="+", default=["count", "binary", "log_tf", "tfidf"])
    ap.add_argument("--layer_list", nargs="+", type=int, default=[1, 2, 3])

    # split
    ap.add_argument("--dev_size", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=13)
    ap.add_argument("--strat_bins", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_path, sep="\t", header="infer")

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found")
    if args.data_col not in df.columns:
        raise ValueError(f"Data column '{args.data_col}' not found")

    train_df, dev_df = stratified_split(
        df,
        args.label_col,
        args.dev_size,
        args.split_seed,
        args.strat_bins,
    )

    log.info(f"Split sizes: train={len(train_df)} dev={len(dev_df)}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Device: {device}")

    all_summaries = []
    all_hist_rows = []

    # Grid search
    for bow_variant in args.bow_variants:
        for n_layers in args.layer_list:
            res = train_one_config(
                args=args,
                train_df=train_df,
                dev_df=dev_df,
                device=device,
                bow_variant=bow_variant,
                n_layers=n_layers,
                run_seed=args.split_seed,
                out_dir=args.out_dir,
            )

            summary = {
                "config_name": res["config_name"],
                "checkpoint": res["checkpoint"],
                "best_epoch": res["best_epoch"],
                "best_dev_rmse": res["best_dev_rmse"],
                **{f"train_{k}": v for k, v in res["final_train_metrics"].items()},
                **{f"dev_{k}": v for k, v in res["final_dev_metrics"].items()},
            }

            all_summaries.append(summary)
            all_hist_rows.extend(res["history"])

            log.info(json.dumps(summary, indent=2))

    # Save summaries
    with open(os.path.join(args.out_dir, "bow_summaries.jsonl"), "w", encoding="utf-8") as f:
        for row in all_summaries:
            f.write(json.dumps(row) + "\n")

    # Save epoch history
    with open(os.path.join(args.out_dir, "bow_history.jsonl"), "w", encoding="utf-8") as f:
        for row in all_hist_rows:
            f.write(json.dumps(row) + "\n")

    # Save best config
    best = min(all_summaries, key=lambda r: r["dev_rmse"])
    with open(os.path.join(args.out_dir, "BEST_BOW.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    log.info(f"Best BoW config by dev RMSE: {best['config_name']}")
    log.info(f"Checkpoint: {best['checkpoint']}")


if __name__ == "__main__":
    main()
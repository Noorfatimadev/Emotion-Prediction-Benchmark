import re
import json
import math
import random
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x).squeeze(-1)
        y_pred.append(out.detach().cpu().numpy())
        y_true.append(y.squeeze(-1).detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        pear, spear = float("nan"), float("nan")
    else:
        pear = pearsonr(y_true, y_pred)[0]
        spear = spearmanr(y_true, y_pred)[0]

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pear),
        "spearman": float(spear),
    }


def stratified_split(df: pd.DataFrame, label_col: str, dev_size: float, seed: int, bins: int = 10):
    y = df[label_col].values
    ranks = pd.Series(y).rank(method="average").values
    strat = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    tr, dv = train_test_split(df, test_size=dev_size, random_state=seed, stratify=strat)
    return tr.reset_index(drop=True), dv.reset_index(drop=True)


def parse_config_name(name: str):
    # bow_<data_col>_<variant>_layers<L>_seed<S>
    m = re.match(r"^bow_(text|lemma|PoS)_(count|binary|log_tf|tfidf)_layers(\d+)_seed(\d+)$", name)
    if not m:
        raise ValueError(f"Unexpected config_name: {name}")
    data_col = m.group(1)
    variant = m.group(2)
    n_layers = int(m.group(3))
    return data_col, variant, n_layers


def mean_std(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
        }
    return out


def train_fixed_epochs(cfg, train_df, dev_df, device, epochs: int, seed: int):
    set_seed(seed)

    train_ds = BowDataset(
        train_df,
        cfg.bow_vocab_size,
        cfg.data_col,
        split="train",
        lowercase=cfg.lowercase,
        vocab=None,
        bow_variant=cfg.bow_variant,
        idf=None,
        label_col=cfg.label_col,
    )
    vocab = train_ds.vocab
    idf = train_ds.get_idf() if cfg.bow_variant == "tfidf" else None

    dev_ds = BowDataset(
        dev_df,
        cfg.bow_vocab_size,
        cfg.data_col,
        split="dev",
        lowercase=cfg.lowercase,
        vocab=vocab,
        bow_variant=cfg.bow_variant,
        idf=idf,
        label_col=cfg.label_col,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MLPModel(cfg, input_dim=cfg.bow_vocab_size, output_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    return evaluate(model, dev_loader, device)


def main():
    ap = ArgumentParser()
    ap.add_argument("--best_json", default="runs_bow/BEST_BOW.json")
    ap.add_argument("--data_path", default="emo_dataset.tsv")
    ap.add_argument("--label_col", default="normalized")
    ap.add_argument("--lowercase", action="store_true", default=True)

    ap.add_argument("--bow_vocab_size", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--dev_size", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=13)
    ap.add_argument("--strat_bins", type=int, default=10)

    ap.add_argument("--repeat_seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out_path", default="BEST_BOW_3RUNS_DEV_MEANSTD.json")
    args = ap.parse_args()

    best = json.load(open(args.best_json, "r", encoding="utf-8"))
    data_col, variant, n_layers = parse_config_name(best["config_name"])
    epochs = int(best["best_epoch"])

    df = pd.read_csv(args.data_path, sep="\t")
    train_df, dev_df = stratified_split(df, args.label_col, args.dev_size, args.split_seed, args.strat_bins)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = deepcopy(args)
    cfg.bow = True
    cfg.data_col = data_col
    cfg.bow_variant = variant
    cfg.n_layers = n_layers

    dev_runs = []
    for s in args.repeat_seeds:
        dev_runs.append(train_fixed_epochs(cfg, train_df, dev_df, device, epochs=epochs, seed=s))

    result = {
        "best_json": args.best_json,
        "config_name": best["config_name"],
        "data_col": data_col,
        "bow_variant": variant,
        "n_layers": n_layers,
        "epochs": epochs,
        "repeat_seeds": args.repeat_seeds,
        "dev_each": dev_runs,
        "dev_mean_std": mean_std(dev_runs),
    }

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
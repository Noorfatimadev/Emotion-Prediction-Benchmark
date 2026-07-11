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

from load_embedding import load_embedding
from datasets1 import EmbeddingsDataset
from data_loading import CollateFunctor
from model import EmbeddingModel


def set_seed(seed: int):
    """Reproducible runs"""
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

    # Correlations break if either vector is (near) constant
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
    """Regression stratification: bin labels by rank and stratify on the bins."""
    y = df[label_col].values
    ranks = pd.Series(y).rank(method="average").values
    strat = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")

    train_df, dev_df = train_test_split(
        df,
        test_size=dev_size,
        random_state=seed,
        stratify=strat,
    )
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def save_model(path: str, args, model: nn.Module):
    torch.save(
        {
            "args": args,
            "model": model.state_dict(),
            "vocab": None,  # not used for embedding-based models
        },
        path,
    )


# Training

def train_one_config(
    args,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    device,
    emb_path: str,
    pooling: str,
    oov_strategy: str,
    finetune: bool,
    n_layers: int,
    seed: int,
    out_dir: str,
):
    set_seed(seed)

    cfg = deepcopy(args)
    cfg.embeddings = emb_path
    cfg.pooling = pooling
    cfg.oov_strategy = oov_strategy
    cfg.train_embeddings = finetune
    cfg.n_layers = n_layers
    cfg.seed = seed
    cfg.bow = False  # used by downstream branching

    # Load pretrained vectors
    word2vec = load_embedding(cfg.embeddings)

    # Special tokens exist for padding and unknown words
    word2vec["[UNK]"] = torch.tensor(word2vec.vectors).mean(dim=0).numpy()
    word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()
    pad_idx = word2vec.get_index("[PAD]")

    train_ds = EmbeddingsDataset(
        train_df,
        word2vec,
        cfg.data_col,
        cfg.lowercase,
        oov_strategy=cfg.oov_strategy,
        label_col=cfg.label_col,
    )
    dev_ds = EmbeddingsDataset(
        dev_df,
        word2vec,
        cfg.data_col,
        cfg.lowercase,
        oov_strategy=cfg.oov_strategy,
        label_col=cfg.label_col,
    )

    collate = CollateFunctor(pad_idx=pad_idx, max_length=cfg.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,  # avoids Windows multiprocessing issues
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = EmbeddingModel(cfg, word2vec, n_labels=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        # Model selection by dev RMSE
        dev_metrics_epoch = evaluate(model, dev_loader, device)
        if dev_metrics_epoch["rmse"] < best_rmse:
            best_rmse = dev_metrics_epoch["rmse"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    cfg_name = (
        f"emb_{os.path.basename(emb_path)}_"
        f"{pooling}_oov{oov_strategy}_"
        f"ft{int(finetune)}_L{n_layers}_seed{seed}"
    )
    ckpt_path = os.path.join(out_dir, cfg_name + ".bin")
    save_model(ckpt_path, cfg, model)

    train_metrics = evaluate(model, train_loader, device)
    dev_metrics = evaluate(model, dev_loader, device)

    return {
        "config_name": cfg_name,
        "checkpoint": ckpt_path,
        "best_epoch": int(best_epoch),
        "best_dev_rmse": float(best_rmse),

        "train_mse": float(train_metrics["mse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_rmse": float(train_metrics["rmse"]),
        "train_pearson": float(train_metrics["pearson"]),
        "train_spearman": float(train_metrics["spearman"]),

        "dev_mse": float(dev_metrics["mse"]),
        "dev_mae": float(dev_metrics["mae"]),
        "dev_rmse": float(dev_metrics["rmse"]),
        "dev_pearson": float(dev_metrics["pearson"]),
        "dev_spearman": float(dev_metrics["spearman"]),
    }


# -------------------- Main --------------------

def main():
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
    )
    log = logging.getLogger("run_embedding_experiments")

    ap = ArgumentParser()

    ap.add_argument("--data_path", default="emo_dataset.tsv")
    ap.add_argument("--data_col", default="lemma", help="text | lemma | PoS")
    ap.add_argument("--label_col", default="normalized", help="Gold label column")
    ap.add_argument("--lowercase", action="store_true", default=True)

    ap.add_argument(
        "--emb_paths",
        nargs="+",
        default=[r"glove.6B.300d.txt", r"embeddings/wikitext103_w2v_300d.bin"],
    )

    ap.add_argument("--poolings", nargs="+", default=["mean", "sum", "max"])
    ap.add_argument("--oov_strategies", nargs="+", default=["unk", "skip"])
    ap.add_argument("--finetune_options", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--layer_list", nargs="+", type=int, default=[1, 2])

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_length", type=int, default=128)

    ap.add_argument("--dev_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--out_dir", default="runs_emb")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_path, sep="\t")
    if args.label_col not in df.columns:
        raise ValueError(f"label_col='{args.label_col}' not found. Columns: {list(df.columns)}")
    if args.data_col not in df.columns:
        raise ValueError(f"data_col='{args.data_col}' not found. Columns: {list(df.columns)}")

    train_df, dev_df = stratified_split(df, args.label_col, args.dev_size, args.seed)
    log.info(f"Split sizes: train={len(train_df)} dev={len(dev_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    for emb in args.emb_paths:
        emb_base = os.path.splitext(os.path.basename(emb))[0]
        summary_path = os.path.join(args.out_dir, f"{emb_base}_summaries.jsonl")

        emb_rows = []
        with open(summary_path, "w", encoding="utf-8") as f:
            for pooling in args.poolings:
                for oov in args.oov_strategies:
                    for ft in args.finetune_options:
                        for layers in args.layer_list:
                            row = train_one_config(
                                args,
                                train_df,
                                dev_df,
                                device,
                                emb,
                                pooling,
                                oov,
                                bool(ft),
                                layers,
                                args.seed,
                                args.out_dir,
                            )
                            emb_rows.append(row)

                            log.info(json.dumps(row, indent=2))
                            f.write(json.dumps(row) + "\n")

        best = min(emb_rows, key=lambda r: r["dev_rmse"])
        best_path = os.path.join(args.out_dir, f"BEST_{emb_base}.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        log.info(f"Saved summaries to: {summary_path}")
        log.info(f"Best config for {emb}: {best['config_name']}")
        log.info(f"Checkpoint: {best['checkpoint']}")
        log.info(f"Saved best config to: {best_path}")


if __name__ == "__main__":
    main()
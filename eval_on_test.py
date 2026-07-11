import tqdm
import pandas as pd
import torch
import torch.nn as nn
import logging
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from load_embedding import load_embedding

# TODO: you have to implement these classes and objects:
from data_loading import CollateFunctor
from datasets1 import BowDataset, EmbeddingsDataset
from model import MLPModel, EmbeddingModel


@torch.no_grad()
def evaluate(regressor: nn.Module, data_iter: DataLoader, comp_device):
    """
    Evaluate regression model on valence prediction.
    Returns MSE, MAE, RMSE, Pearson correlation, and Spearman correlation.
    """
    regressor.eval()
    labels_true, predictions = [], []
    for batch in tqdm.tqdm(data_iter):
        input_ids, label_true = (t.to(comp_device) for t in batch)
        output = regressor(input_ids)
        predictions += output.squeeze().tolist()
        labels_true += label_true.squeeze().tolist()

    # Convert to numpy arrays
    labels_true = np.array(labels_true)
    predictions = np.array(predictions)

    # Calculate regression metrics
    mse = mean_squared_error(labels_true, predictions)
    mae = mean_absolute_error(labels_true, predictions)
    rmse = np.sqrt(mse)

    # Calculate correlation coefficients
    pearson_corr, _ = pearsonr(labels_true, predictions)
    spearman_corr, _ = spearmanr(labels_true, predictions)

    return mse, mae, rmse, pearson_corr, spearman_corr


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    # add command line arguments
    # this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    parser.add_argument(
        "--test_path",
        action="store",
        help="Path to test/dev data TSV file",
        default="emo_test.tsv",
    )
    parser.add_argument("--model", default="runs_emb/emb_wikitext103_w2v_300d.bin_mean_oovskip_ft0_L2_seed13.bin", help="Path to saved model")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    saved_model = torch.load(args.model, map_location=device, weights_only=False)
    bow_idf = saved_model.get("bow_idf", None)
    logger.info(f"Loaded model from {args.model}")
    training_args = saved_model["args"]
    state_dict = saved_model["model"]
    train_vocab = saved_model["vocab"]

    df = pd.read_csv(args.test_path, sep="\t", header="infer")
    logger.info(training_args)

    if training_args.bow:
        logger.info(f"Loading data")
        test_dataset = BowDataset(
            df,
            training_args.bow_vocab_size,
            training_args.data_col,
            "test",
            training_args.lowercase,
            vocab=train_vocab,
            bow_variant=getattr(training_args, "bow_variant", "count"),
            idf=bow_idf,
        )

        test_iter = DataLoader(test_dataset, batch_size=128, shuffle=False)
        logger.info(f"Data loaded")
        model = MLPModel(
            training_args, training_args.bow_vocab_size, output_dim=1
        ).to(device)
    else:
        logger.info(f"Loading vectors from {training_args.embeddings}")
        word2vec = load_embedding(training_args.embeddings)
        word2vec["[UNK]"] = torch.tensor(word2vec.vectors).mean(dim=0).numpy()
        word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()
        logger.info(f"Loading data")
        test_dataset = EmbeddingsDataset(
            df, word2vec, training_args.data_col, training_args.lowercase
        )
        test_iter = DataLoader(
            test_dataset,
            batch_size=training_args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=CollateFunctor(
                word2vec.get_index("[PAD]"), training_args.max_length
            ),
        )
        model = EmbeddingModel(training_args, word2vec, n_labels=1).to(device)

    model.load_state_dict(state_dict)
    model_size = sum(p.numel() for p in model.parameters())

    total_mse, total_mae, total_rmse, pearson, spearman = evaluate(
        model, test_iter, device
    )
    logger.info(f"Model size: {model_size} parameters")
    logger.info(f"=== Regression Metrics ===")
    logger.info(f"MSE (Mean Squared Error): {total_mse:.4f}")
    logger.info(f"MAE (Mean Absolute Error): {total_mae:.4f}")
    logger.info(f"RMSE (Root Mean Squared Error): {total_rmse:.4f}")
    logger.info(f"Pearson Correlation: {pearson:.4f}")
    logger.info(f"Spearman Correlation: {spearman:.4f}")

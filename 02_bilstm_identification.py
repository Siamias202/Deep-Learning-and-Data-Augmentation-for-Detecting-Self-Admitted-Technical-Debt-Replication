"""
02_bilstm_identification.py
============================
BiLSTM-based Binary Identification of SATD (SATD vs Not-SATD).

Paper: "Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt"
       Sutoyo et al., 2024 (arXiv:2410.15804v1)

Paper methodology (Section III-F):
    Architecture:
      - Embedding layer: vocabulary size × embedding dim, pre-trained GloVe matrix
      - BiLSTM layer 1 : 128 units + Dropout(0.3) + Batch Normalization
      - BiLSTM layer 2 : 64  units + Dropout(0.3)
      - BiLSTM layer 3 : 128 units + Dropout(0.3)
      - BiLSTM layer 4 : 128 units  (final, consolidates features)
      - Dense output   : sigmoid activation (binary classification)
    Training:
      - Early stopping on validation loss
      - GloVe embeddings (100-d default)
      - 80/10/10 stratified train/val/test split

Usage examples:
    # Train on CC artifact (code comments):
    python 02_bilstm_identification.py --artifact CC --mode train

    # Train with augmented data:
    python 02_bilstm_identification.py --artifact IS --mode train --use_augmented

    # Evaluate on saved model:
    python 02_bilstm_identification.py --artifact CM --mode evaluate

    # All artifacts, train + evaluate:
    python 02_bilstm_identification.py --artifact all --mode train

Command-line Arguments:
    --artifact        {CC, IS, PS, CM, all}   Artifact to process (default: all)
    --mode            {train, evaluate}        Run mode (default: train)
    --data_dir        Path to preprocessed CSVs (default: data/preprocessed)
    --glove_path      Path to GloVe .txt file  (default: data/glove.6B.100d.txt)
    --embedding_dim   GloVe dimension           (default: 100)
    --max_seq_len     Max token sequence length (default: 200)
    --batch_size      Training batch size       (default: 64)
    --epochs          Max training epochs       (default: 50)
    --patience        Early-stopping patience   (default: 5)
    --use_augmented   Flag: use augmented training data if available
    --output_dir      Directory for results     (default: results)
    --model_dir       Directory for saved models (default: models)

Output:
    - models/bilstm_{artifact}.pt          : Best model checkpoint
    - models/bilstm_{artifact}_vocab.json  : Vocabulary (word→index mapping)
    - models/bilstm_{artifact}_config.json : Model hyperparameters
    - results/bilstm_command_arguments.txt : Results log (appended each run)
"""

import os
import re
import json
import time
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTIFACT_FILES = {
    "CC": "preprocessed_cc.csv",
    "CM": "preprocessed_cm.csv",
    "IS": "preprocessed_is.csv",
    "PS": "preprocessed_ps.csv",
}

LABEL2IDX = {"Not-SATD": 0, "SATD": 1}
IDX2LABEL = {0: "Not-SATD", 1: "SATD"}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SATDDataset(Dataset):
    """PyTorch Dataset for SATD binary identification."""

    def __init__(self, texts: list, labels: list, word2idx: dict, max_seq_len: int):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def _encode(self, text: str) -> list:
        tokens = str(text).split()
        indices = [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in tokens]
        # Truncate or pad to max_seq_len
        if len(indices) >= self.max_seq_len:
            return indices[: self.max_seq_len]
        return indices + [self.word2idx[PAD_TOKEN]] * (self.max_seq_len - len(indices))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self._encode(self.texts[idx])
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# BiLSTM Model (paper Section III-F)
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """
    Stacked Bidirectional LSTM for binary SATD identification.

    Architecture from paper Section III-F:
        Embedding  → BiLSTM(128) → Dropout(0.3) → BatchNorm
                   → BiLSTM(64)  → Dropout(0.3)
                   → BiLSTM(128) → Dropout(0.3)
                   → BiLSTM(128)                  [final]
                   → FC → Sigmoid
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_matrix: np.ndarray,
        num_classes: int = 2,
        pad_idx: int = 0,
    ):
        super(BiLSTMClassifier, self).__init__()

        # Embedding layer (pre-trained GloVe, frozen by default)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = True  # fine-tune embeddings

        # BiLSTM layer 1: 128 units
        self.bilstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)  # 128*2 for bidirectional

        # BiLSTM layer 2: 64 units
        self.bilstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)

        # BiLSTM layer 3: 128 units
        self.bilstm3 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout3 = nn.Dropout(0.3)

        # BiLSTM layer 4: 128 units (final, consolidates features)
        self.bilstm4 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Fully connected output
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)

        # Embedding
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)

        # BiLSTM 1
        out1, _ = self.bilstm1(emb)         # (batch, seq_len, 256)
        out1 = self.dropout1(out1)
        # BatchNorm on last time step (paper normalises after first BiLSTM)
        last1 = out1[:, -1, :]              # (batch, 256)
        last1 = self.batchnorm1(last1)
        out1 = out1  # keep full sequence for next layer

        # BiLSTM 2
        out2, _ = self.bilstm2(out1)        # (batch, seq_len, 128)
        out2 = self.dropout2(out2)

        # BiLSTM 3
        out3, _ = self.bilstm3(out2)        # (batch, seq_len, 256)
        out3 = self.dropout3(out3)

        # BiLSTM 4 (final)
        out4, _ = self.bilstm4(out3)        # (batch, seq_len, 256)
        last4 = out4[:, -1, :]             # take last time step (batch, 256)

        # Classification head
        logits = self.fc(last4)            # (batch, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Vocabulary & GloVe
# ---------------------------------------------------------------------------

def build_vocabulary(texts: list, min_freq: int = 1) -> dict:
    """Build word-to-index mapping from training corpus."""
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    logger.info(f"Vocabulary size: {len(word2idx):,}")
    return word2idx


def load_glove_embeddings(glove_path: str, word2idx: dict, embedding_dim: int) -> np.ndarray:
    """
    Load GloVe pre-trained vectors and build embedding matrix.
    Words not found in GloVe are randomly initialised.
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(
        np.float32
    )
    # PAD → zero vector
    embedding_matrix[0] = np.zeros(embedding_dim, dtype=np.float32)

    if not os.path.isfile(glove_path):
        logger.warning(
            f"GloVe file not found at '{glove_path}'. "
            "Using randomly initialised embeddings instead."
        )
        return embedding_matrix

    logger.info(f"Loading GloVe embeddings from: {glove_path}")
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            word = parts[0]
            if word in word2idx:
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == embedding_dim:
                        embedding_matrix[word2idx[word]] = vector
                        found += 1
                except ValueError:
                    pass

    coverage = found / max(vocab_size - 2, 1) * 100
    logger.info(f"GloVe coverage: {found:,}/{vocab_size-2:,} vocab words ({coverage:.1f}%)")
    return embedding_matrix


def save_vocab(word2idx: dict, path: str):
    """Persist vocabulary to JSON (needed for prediction pipeline)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    logger.info(f"Vocabulary saved → {path}")


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_config(config: dict, path: str):
    """Persist model hyperparameters to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Model config saved → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_artifact_data(
    artifact_key: str,
    data_dir: str,
    use_augmented: bool = False,
) -> pd.DataFrame:
    """Load preprocessed CSV for the given artifact."""
    filename = ARTIFACT_FILES[artifact_key]
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}\n"
            "Run 01_preprocessing.py first."
        )
    df = pd.read_csv(path, encoding="utf-8")

    required_cols = {"cleaned_text", "binary_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    logger.info(f"[{artifact_key}] Loaded {len(df):,} rows from {path}")

    if use_augmented:
        aug_path = os.path.join(
            data_dir,
            "..",  # go up to data/
            f"augmented_{artifact_key.lower()}.csv",
        )
        aug_path = os.path.normpath(aug_path)
        if os.path.isfile(aug_path):
            df_aug = pd.read_csv(aug_path, encoding="utf-8")
            logger.info(f"[{artifact_key}] Loaded {len(df_aug):,} augmented rows from {aug_path}")
            df = pd.concat([df, df_aug], ignore_index=True)
        else:
            logger.warning(
                f"[{artifact_key}] --use_augmented set but augmented file not found: {aug_path}"
            )

    return df


def stratified_split(df: pd.DataFrame, label_col: str = "binary_label"):
    """
    80/10/10 stratified train/val/test split as per paper Section III-H.
    """
    X = df["cleaned_text"].tolist()
    y = df[label_col].map(LABEL2IDX).tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(
        f"Split → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, all_preds, all_labels


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute per-class and macro-averaged F1, Precision, Recall."""
    report = classification_report(
        y_true, y_pred,
        target_names=[IDX2LABEL[0], IDX2LABEL[1]],
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"report": report, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

def append_result(result_path: str, content: str):
    """Append a result block to the shared result text file."""
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")


def format_result_block(
    artifact: str,
    args: argparse.Namespace,
    metrics: dict,
    elapsed: float,
) -> str:
    """Format a results block for the output text file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = metrics["report"]

    lines = [
        "=" * 70,
        f"BiLSTM SATD Identification Results",
        f"Timestamp   : {ts}",
        f"Artifact    : {artifact}",
        "Command Arguments:",
        f"  --artifact      {args.artifact}",
        f"  --mode          {args.mode}",
        f"  --data_dir      {args.data_dir}",
        f"  --glove_path    {args.glove_path}",
        f"  --embedding_dim {args.embedding_dim}",
        f"  --max_seq_len   {args.max_seq_len}",
        f"  --batch_size    {args.batch_size}",
        f"  --epochs        {args.epochs}",
        f"  --patience      {args.patience}",
        f"  --use_augmented {args.use_augmented}",
        f"  --output_dir    {args.output_dir}",
        f"  --model_dir     {args.model_dir}",
        "-" * 70,
        "Test Set Performance:",
        f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        f"  {'-'*55}",
    ]

    for cls in ["Not-SATD", "SATD"]:
        if cls in report:
            r = report[cls]
            lines.append(
                f"  {cls:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1-score']:>10.4f} {int(r['support']):>10}"
            )

    lines += [
        f"  {'-'*55}",
        f"  {'Macro Avg':<15} {report['macro avg']['precision']:>10.4f} "
        f"{report['macro avg']['recall']:>10.4f} "
        f"{metrics['macro_f1']:>10.4f}",
        f"  {'Accuracy':<15} {report['accuracy']:>10.4f}",
        f"Training time : {elapsed:.1f}s",
        "=" * 70,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_artifact(artifact_key: str, args: argparse.Namespace, result_path: str):
    """Full train + evaluate pipeline for one artifact."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing artifact: {artifact_key}")
    logger.info(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_artifact_data(artifact_key, args.data_dir, args.use_augmented)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    # ── Build vocabulary from training set only (prevent data leakage) ───
    word2idx = build_vocabulary(X_train)

    # ── Load GloVe embeddings ─────────────────────────────────────────────
    embedding_matrix = load_glove_embeddings(
        args.glove_path, word2idx, args.embedding_dim
    )

    # ── Save vocabulary & config ──────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    vocab_path = os.path.join(args.model_dir, f"bilstm_{artifact_key}_vocab.json")
    config_path = os.path.join(args.model_dir, f"bilstm_{artifact_key}_config.json")
    model_path = os.path.join(args.model_dir, f"bilstm_{artifact_key}.pt")

    save_vocab(word2idx, vocab_path)
    config = {
        "vocab_size": len(word2idx),
        "embedding_dim": args.embedding_dim,
        "max_seq_len": args.max_seq_len,
        "num_classes": 2,
        "pad_idx": word2idx[PAD_TOKEN],
        "label2idx": LABEL2IDX,
        "artifact": artifact_key,
    }
    save_model_config(config, config_path)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = SATDDataset(X_train, y_train, word2idx, args.max_seq_len)
    val_ds   = SATDDataset(X_val,   y_val,   word2idx, args.max_seq_len)
    test_ds  = SATDDataset(X_test,  y_test,  word2idx, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        embedding_matrix=embedding_matrix,
        num_classes=2,
        pad_idx=word2idx[PAD_TOKEN],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Training loop with early stopping on validation loss ──────────────
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    logger.info(f"[{artifact_key}] Training on {DEVICE} for up to {args.epochs} epochs …")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        logger.info(
            f"[{artifact_key}] Epoch {epoch:03d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f} TrainAcc: {train_acc:.4f} | "
            f"ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}"
        )

        # Early stopping: save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"[{artifact_key}] Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"[{artifact_key}] Early stopping triggered at epoch {epoch} "
                    f"(patience={args.patience})"
                )
                break

    elapsed = time.time() - start_time

    # ── Test evaluation ───────────────────────────────────────────────────
    logger.info(f"[{artifact_key}] Loading best model for test evaluation …")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(f"\n[{artifact_key}] TEST RESULTS (Macro F1 = {metrics['macro_f1']:.4f}):")
    logger.info(
        classification_report(
            y_true, y_pred,
            target_names=[IDX2LABEL[0], IDX2LABEL[1]],
            zero_division=0,
        )
    )

    # ── Write results to file ─────────────────────────────────────────────
    result_block = format_result_block(artifact_key, args, metrics, elapsed)
    append_result(result_path, result_block)
    logger.info(f"[{artifact_key}] Results appended → {result_path}")

    return metrics


def run_evaluate_only(artifact_key: str, args: argparse.Namespace, result_path: str):
    """Load saved model and evaluate on the test set."""
    model_path  = os.path.join(args.model_dir, f"bilstm_{artifact_key}.pt")
    vocab_path  = os.path.join(args.model_dir, f"bilstm_{artifact_key}_vocab.json")
    config_path = os.path.join(args.model_dir, f"bilstm_{artifact_key}_config.json")

    for p in (model_path, vocab_path, config_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}. Run in train mode first.")

    word2idx = load_vocab(vocab_path)
    with open(config_path) as f:
        config = json.load(f)

    df = load_artifact_data(artifact_key, args.data_dir, use_augmented=False)
    _, _, X_test, _, _, y_test = stratified_split(df)

    test_ds     = SATDDataset(X_test, y_test, word2idx, config["max_seq_len"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        embedding_matrix=None,
        num_classes=config["num_classes"],
        pad_idx=config["pad_idx"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    start = time.time()
    _, _, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
    elapsed = time.time() - start

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"\n[{artifact_key}] EVALUATION RESULTS (Macro F1 = {metrics['macro_f1']:.4f}):")
    logger.info(
        classification_report(
            y_true, y_pred,
            target_names=[IDX2LABEL[0], IDX2LABEL[1]],
            zero_division=0,
        )
    )

    result_block = format_result_block(artifact_key, args, metrics, elapsed)
    append_result(result_path, result_block)
    logger.info(f"[{artifact_key}] Results appended → {result_path}")
    return metrics


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="BiLSTM SATD Identification (Sutoyo et al., 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="all",
        choices=["CC", "IS", "PS", "CM", "all"],
        help="Artifact to process. 'all' runs all four.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="'train' trains and evaluates; 'evaluate' loads saved model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preprocessed"),
        help="Directory containing preprocessed CSV files.",
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "glove.6B.100d.txt"),
        help="Path to GloVe embeddings text file.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="Dimensionality of GloVe embeddings.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=200,
        help="Maximum token sequence length (pad/truncate).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience (epochs without val loss improvement).",
    )
    parser.add_argument(
        "--use_augmented",
        action="store_true",
        help="Include AugGPT-augmented training samples if available.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Directory to save result text files.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Directory to save/load model checkpoints, vocab, and configs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Result file (appended per run, as required)
    result_path = os.path.join(args.output_dir, "bilstm_command_arguments.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Determine which artifacts to run
    if args.artifact == "all":
        artifacts = list(ARTIFACT_FILES.keys())
    else:
        artifacts = [args.artifact]

    all_metrics = {}
    for artifact_key in artifacts:
        try:
            if args.mode == "train":
                metrics = run_artifact(artifact_key, args, result_path)
            else:
                metrics = run_evaluate_only(artifact_key, args, result_path)
            all_metrics[artifact_key] = metrics
        except FileNotFoundError as e:
            logger.error(str(e))
        except Exception as e:
            logger.exception(f"[{artifact_key}] Unexpected error: {e}")

    # Summary across all processed artifacts
    if all_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL SUMMARY (Macro F1-scores)")
        logger.info("=" * 60)
        for art, m in all_metrics.items():
            logger.info(f"  {art:4s} → Macro F1 = {m['macro_f1']:.4f}")
        logger.info(f"\nFull results saved to: {result_path}")


if __name__ == "__main__":
    main()
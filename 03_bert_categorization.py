"""
03_bert_classification.py
==========================
BERT-based Multi-class Categorization of SATD Types.

Paper: "Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt"
       Sutoyo et al., 2024 (arXiv:2410.15804v1)

Paper methodology (Section III-G):
    Model : bert-base-uncased (768 hidden units, 12 layers, 110M parameters)
    Classifier head:
        Linear(768 → hidden_size) → ReLU → Linear(hidden_size → num_classes)
    Loss       : CrossEntropyLoss
    Optimizer  : AdamW (weight decay)
    LR         : 5e-5
    Epsilon    : 1e-8
    Batch size : 32
    Classes    : C/D (code/design), DOC (documentation), TES (test), REQ (requirement)

    Input: ONLY items already identified as SATD (from BiLSTM step or ground truth).
    Output: One of {C/D, DOC, TES, REQ}

    Evaluation: Stratified 80/10/10 train/val/test split, F1-score.

Usage examples:
    # Train on IS artifact (issues):
    python 03_bert_classification.py --artifact IS --mode train

    # Train with AugGPT-augmented data:
    python 03_bert_classification.py --artifact CC --mode train --use_augmented

    # Evaluate saved model:
    python 03_bert_classification.py --artifact PS --mode evaluate

    # All artifacts:
    python 03_bert_classification.py --artifact all --mode train

Command-line Arguments:
    --artifact        {CC, IS, PS, CM, all}   Artifact to process (default: all)
    --mode            {train, evaluate}        Run mode (default: train)
    --data_dir        Path to preprocessed CSVs (default: data/preprocessed)
    --bert_model      HuggingFace model name   (default: bert-base-uncased)
    --max_seq_len     Max token length          (default: 128)
    --batch_size      Training batch size       (default: 32)
    --epochs          Max training epochs       (default: 10)
    --learning_rate   AdamW learning rate       (default: 5e-5)
    --epsilon         AdamW epsilon             (default: 1e-8)
    --hidden_size     Classifier hidden layer   (default: 256)
    --patience        Early-stopping patience   (default: 3)
    --use_augmented   Flag: include augmented training samples
    --output_dir      Results directory         (default: results)
    --model_dir       Model checkpoint dir      (default: models)

Output:
    - models/bert_{artifact}/          : Saved BERT model + tokenizer
    - models/bert_{artifact}_config.json : Model configuration
    - results/bert_command_arguments.txt : Results log (appended each run)
"""

import os
import json
import time
import logging
import argparse
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
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

# Multi-class SATD categories (paper Section III-G)
SATD_CATEGORIES = ["C/D", "DOC", "TES", "REQ"]
LABEL2IDX = {label: idx for idx, label in enumerate(SATD_CATEGORIES)}
IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}
NOT_SATD_LABEL = "Not-SATD"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SATDCategoryDataset(Dataset):
    """
    PyTorch Dataset for BERT-based SATD categorization.
    Only SATD instances are included (Not-SATD filtered out).
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: BertTokenizer,
        max_seq_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids",
                torch.zeros_like(encoding["input_ids"]),
            ).squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# BERT Classifier (paper Section III-G)
# ---------------------------------------------------------------------------

class BERTSATDClassifier(nn.Module):
    """
    BERT-base-uncased with a two-layer classification head.

    Head architecture (paper Section III-G):
        Linear(768 → hidden_size) → ReLU → Linear(hidden_size → num_classes)
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        hidden_size: int = 256,
        dropout_rate: float = 0.1,
    ):
        super(BERTSATDClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Classifier head: Linear → ReLU → Linear (paper architecture)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output  # (batch, 768)
        logits = self.classifier(pooled_output)  # (batch, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_artifact_data(
    artifact_key: str,
    data_dir: str,
    use_augmented: bool = False,
) -> pd.DataFrame:
    """
    Load preprocessed CSV, filter to SATD-only rows, and resolve labels.
    Only rows with valid SATD category labels (C/D, DOC, TES, REQ) are kept.
    """
    filename = ARTIFACT_FILES[artifact_key]
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}\n"
            "Run 01_preprocessing.py first."
        )
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"[{artifact_key}] Loaded {len(df):,} total rows from {path}")

    # ── Keep only SATD instances (filter out Not-SATD) ────────────────────
    # The 'label' column contains the multi-class SATD type
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {path}")

    # Filter to known SATD categories only
    df_satd = df[df["label"].isin(SATD_CATEGORIES)].copy()
    logger.info(
        f"[{artifact_key}] After SATD filter: {len(df_satd):,} rows "
        f"(removed {len(df) - len(df_satd):,} Not-SATD rows)"
    )

    if use_augmented:
        aug_path = os.path.normpath(
            os.path.join(data_dir, "..", f"augmented_{artifact_key.lower()}.csv")
        )
        if os.path.isfile(aug_path):
            df_aug = pd.read_csv(aug_path, encoding="utf-8")
            df_aug = df_aug[df_aug["label"].isin(SATD_CATEGORIES)].copy()
            logger.info(
                f"[{artifact_key}] Augmented SATD rows: {len(df_aug):,} from {aug_path}"
            )
            df_satd = pd.concat([df_satd, df_aug], ignore_index=True)
        else:
            logger.warning(
                f"[{artifact_key}] --use_augmented set but augmented file not found: {aug_path}"
            )

    # ── Label distribution ────────────────────────────────────────────────
    logger.info(f"[{artifact_key}] SATD category distribution:")
    for cat, cnt in df_satd["label"].value_counts().items():
        logger.info(f"           {cat:10s}: {cnt:6d}")

    return df_satd


def stratified_split(df: pd.DataFrame):
    """
    80/10/10 stratified split on SATD categories (paper Section III-H).
    Uses original_text (or cleaned_text if original not present) as input.
    """
    text_col = "original_text" if "original_text" in df.columns else "cleaned_text"
    X = df[text_col].tolist()
    y = df["label"].map(LABEL2IDX).tolist()

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

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    return avg_loss, acc, all_preds, all_labels


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute per-class and macro-averaged F1 (as per paper)."""
    # Only include classes that actually appear in test set
    present_labels = sorted(set(y_true))
    target_names   = [IDX2LABEL[i] for i in present_labels]

    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"report": report, "macro_f1": macro_f1, "target_names": target_names}


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
    target_names = metrics.get("target_names", SATD_CATEGORIES)

    lines = [
        "=" * 70,
        f"BERT SATD Categorization Results",
        f"Timestamp      : {ts}",
        f"Artifact       : {artifact}",
        "Command Arguments:",
        f"  --artifact      {args.artifact}",
        f"  --mode          {args.mode}",
        f"  --data_dir      {args.data_dir}",
        f"  --bert_model    {args.bert_model}",
        f"  --max_seq_len   {args.max_seq_len}",
        f"  --batch_size    {args.batch_size}",
        f"  --epochs        {args.epochs}",
        f"  --learning_rate {args.learning_rate}",
        f"  --epsilon       {args.epsilon}",
        f"  --hidden_size   {args.hidden_size}",
        f"  --patience      {args.patience}",
        f"  --use_augmented {args.use_augmented}",
        f"  --output_dir    {args.output_dir}",
        f"  --model_dir     {args.model_dir}",
        "-" * 70,
        "Test Set Performance (SATD categories):",
        f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        f"  {'-'*55}",
    ]

    for cls in target_names:
        if cls in report:
            r = report[cls]
            lines.append(
                f"  {cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1-score']:>10.4f} {int(r['support']):>10}"
            )

    lines += [
        f"  {'-'*55}",
        f"  {'Macro Avg':<12} {report['macro avg']['precision']:>10.4f} "
        f"{report['macro avg']['recall']:>10.4f} "
        f"{metrics['macro_f1']:>10.4f}",
    ]
    if "accuracy" in report:
        lines.append(f"  {'Accuracy':<12} {report['accuracy']:>10.4f}")

    lines += [
        f"Training time  : {elapsed:.1f}s",
        "=" * 70,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_artifact(artifact_key: str, args: argparse.Namespace, result_path: str):
    """Full train + evaluate pipeline for one artifact."""
    logger.info(f"\n{'='*60}")
    logger.info(f"BERT Categorization — Artifact: {artifact_key}")
    logger.info(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_artifact_data(artifact_key, args.data_dir, args.use_augmented)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info(f"[{artifact_key}] Loading tokenizer: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = SATDCategoryDataset(X_train, y_train, tokenizer, args.max_seq_len)
    val_ds   = SATDCategoryDataset(X_val,   y_val,   tokenizer, args.max_seq_len)
    test_ds  = SATDCategoryDataset(X_test,  y_test,  tokenizer, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info(f"[{artifact_key}] Loading BERT model: {args.bert_model}")
    model = BERTSATDClassifier(
        bert_model_name=args.bert_model,
        num_classes=len(SATD_CATEGORIES),
        hidden_size=args.hidden_size,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # AdamW optimizer with paper hyperparameters (lr=5e-5, eps=1e-8)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.epsilon,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ── Save paths ────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model_save_dir = os.path.join(args.model_dir, f"bert_{artifact_key}")
    config_path    = os.path.join(args.model_dir, f"bert_{artifact_key}_config.json")
    best_model_path = os.path.join(args.model_dir, f"bert_{artifact_key}_best.pt")

    # Save config for inference pipeline
    config = {
        "bert_model": args.bert_model,
        "max_seq_len": args.max_seq_len,
        "num_classes": len(SATD_CATEGORIES),
        "hidden_size": args.hidden_size,
        "label2idx": LABEL2IDX,
        "idx2label": {str(k): v for k, v in IDX2LABEL.items()},
        "artifact": artifact_key,
        "categories": SATD_CATEGORIES,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"[{artifact_key}] Config saved → {config_path}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    logger.info(
        f"[{artifact_key}] Training on {DEVICE} | "
        f"lr={args.learning_rate} | batch={args.batch_size} | max_epochs={args.epochs}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        logger.info(
            f"[{artifact_key}] Epoch {epoch:03d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f} TrainAcc: {train_acc:.4f} | "
            f"ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # Also save tokenizer alongside model
            os.makedirs(model_save_dir, exist_ok=True)
            tokenizer.save_pretrained(model_save_dir)
            logger.info(
                f"[{artifact_key}] Best model saved (val_loss={best_val_loss:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"[{artifact_key}] Early stopping at epoch {epoch} "
                    f"(patience={args.patience})"
                )
                break

    elapsed = time.time() - start_time

    # ── Test evaluation ───────────────────────────────────────────────────
    logger.info(f"[{artifact_key}] Loading best model for test evaluation …")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(f"\n[{artifact_key}] TEST RESULTS (Macro F1 = {metrics['macro_f1']:.4f}):")
    logger.info(
        classification_report(
            y_true, y_pred,
            labels=sorted(set(y_true)),
            target_names=metrics["target_names"],
            zero_division=0,
        )
    )

    result_block = format_result_block(artifact_key, args, metrics, elapsed)
    append_result(result_path, result_block)
    logger.info(f"[{artifact_key}] Results appended → {result_path}")

    return metrics


def run_evaluate_only(artifact_key: str, args: argparse.Namespace, result_path: str):
    """Load saved BERT model and evaluate on the test set."""
    config_path     = os.path.join(args.model_dir, f"bert_{artifact_key}_config.json")
    best_model_path = os.path.join(args.model_dir, f"bert_{artifact_key}_best.pt")
    model_save_dir  = os.path.join(args.model_dir, f"bert_{artifact_key}")

    for p in (config_path, best_model_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}. Run in train mode first.")

    with open(config_path) as f:
        config = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(
        model_save_dir if os.path.isdir(model_save_dir) else config["bert_model"]
    )

    df = load_artifact_data(artifact_key, args.data_dir, use_augmented=False)
    _, _, X_test, _, _, y_test = stratified_split(df)

    test_ds     = SATDCategoryDataset(X_test, y_test, tokenizer, config["max_seq_len"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BERTSATDClassifier(
        bert_model_name=config["bert_model"],
        num_classes=config["num_classes"],
        hidden_size=config["hidden_size"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    start = time.time()
    _, _, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
    elapsed = time.time() - start

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"\n[{artifact_key}] EVALUATION RESULTS (Macro F1 = {metrics['macro_f1']:.4f}):")
    logger.info(
        classification_report(
            y_true, y_pred,
            labels=sorted(set(y_true)),
            target_names=metrics["target_names"],
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
        description="BERT SATD Categorization (Sutoyo et al., 2024)",
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
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace BERT model identifier.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Maximum BERT tokenizer sequence length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size (paper: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="AdamW learning rate (paper: 5e-5).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="AdamW epsilon (paper: 1e-8).",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Size of the hidden layer in the BERT classification head.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stopping patience epochs.",
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
        help="Directory to save/load BERT model checkpoints and configs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    result_path = os.path.join(args.output_dir, "bert_command_arguments.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

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

    if all_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("BERT CATEGORIZATION SUMMARY (Macro F1)")
        logger.info("=" * 60)
        for art, m in all_metrics.items():
            logger.info(f"  {art:4s} → Macro F1 = {m['macro_f1']:.4f}")
        logger.info(f"\nFull results saved to: {result_path}")


if __name__ == "__main__":
    main()
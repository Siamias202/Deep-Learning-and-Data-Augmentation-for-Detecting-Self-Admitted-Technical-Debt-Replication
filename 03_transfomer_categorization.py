"""
03_transformer_classification.py
================================

Transformer-based Multi-class Categorization of SATD Types.

Supports:
    - BERT
    - RoBERTa
    - DeBERTa
    - DistilBERT
    - Other HuggingFace encoder models compatible with AutoModel

Paper:
    "Deep Learning and Data Augmentation for Detecting Self-Admitted
     Technical Debt"
    Sutoyo et al., 2024
    arXiv:2410.15804v1

Task:
    Input:
        ONLY SATD instances

    Output:
        documentation_debt
        requirement_debt
        test_debt
        structural_debt

Architecture:
    Transformer encoder
        ↓
    Mean pooling
        ↓
    Linear(hidden_size → classifier_hidden)
        ↓
    ReLU
        ↓
    Dropout
        ↓
    Linear(classifier_hidden → 4)

Usage:

    # BERT
    python 03_transformer_classification.py \
        --artifact IS \
        --mode train \
        --bert_model bert-base-uncased

    # RoBERTa
    python 03_transformer_classification.py \
        --artifact IS \
        --mode train \
        --bert_model roberta-base

    # DeBERTa
    python 03_transformer_classification.py \
        --artifact IS \
        --mode train \
        --bert_model microsoft/deberta-v3-base

    # All artifacts
    python 03_transformer_classification.py \
        --artifact all \
        --mode train \
        --bert_model microsoft/deberta-v3-base

    # With augmentation
    python 03_transformer_classification.py \
        --artifact CC \
        --mode train \
        --bert_model microsoft/deberta-v3-base \
        --use_augmented

    # Evaluate saved model
    python 03_transformer_classification.py \
        --artifact IS \
        --mode evaluate

Recommended starting configuration for DeBERTa:
    --max_seq_len 128
    --batch_size 16
    --epochs 10
    --learning_rate 2e-5
    --hidden_size 256
    --patience 3
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
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

ARTIFACT_FILES = {
    "CC": "preprocessed_cc.csv",
    "CM": "preprocessed_cm.csv",
    "IS": "preprocessed_is.csv",
    "PS": "preprocessed_ps.csv",
}


# SATD categories
SATD_CATEGORIES = [
    "documentation_debt",
    "requirement_debt",
    "test_debt",
    "structural_debt",
]


LABEL2IDX = {
    label: idx
    for idx, label in enumerate(SATD_CATEGORIES)
}

IDX2LABEL = {
    idx: label
    for label, idx in LABEL2IDX.items()
}


NOT_SATD_LABEL = "non_debt"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# Random Seed
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset
# ============================================================================

class SATDCategoryDataset(Dataset):
    """
    Dataset for Transformer-based SATD categorization.

    Only SATD instances are included.
    """

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_seq_len,
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "class": torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        }

        # Some models, such as BERT, provide token_type_ids.
        # RoBERTa and DeBERTa generally do not need them.
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding[
                "token_type_ids"
            ].squeeze(0)

        return item


# ============================================================================
# Mean Pooling
# ============================================================================

def mean_pooling(
    last_hidden_state,
    attention_mask,
):
    """
    Attention-mask-aware mean pooling.

    last_hidden_state:
        [batch_size, sequence_length, hidden_size]

    attention_mask:
        [batch_size, sequence_length]
    """

    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    masked_embeddings = (
        last_hidden_state * mask
    )

    summed = torch.sum(
        masked_embeddings,
        dim=1,
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


# ============================================================================
# Transformer Classifier
# ============================================================================

class TransformerSATDClassifier(nn.Module):
    """
    Generic Transformer SATD classifier.

    Supports models loaded through HuggingFace AutoModel.

    Examples:

        bert-base-uncased

        bert-large-uncased

        roberta-base

        roberta-large

        microsoft/deberta-v3-base

        microsoft/deberta-v3-large

        distilbert-base-uncased
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        num_classes=4,
        classifier_hidden_size=256,
        dropout_rate=0.1,
    ):

        super().__init__()

        logger.info(
            f"Loading Transformer model: {model_name}"
        )

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        encoder_hidden_size = (
            self.encoder.config.hidden_size
        )

        logger.info(
            f"Encoder hidden size: "
            f"{encoder_hidden_size}"
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                encoder_hidden_size,
                classifier_hidden_size,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                classifier_hidden_size,
                num_classes,
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):

        # --------------------------------------------------------
        # Some Transformer models use token_type_ids.
        # Others, such as RoBERTa/DeBERTa, don't.
        # --------------------------------------------------------

        if token_type_ids is not None:

            try:

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

            except TypeError:

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        else:

            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # --------------------------------------------------------
        # Mean pooling
        # --------------------------------------------------------

        pooled_output = mean_pooling(
            outputs.last_hidden_state,
            attention_mask,
        )

        # --------------------------------------------------------
        # Classification head
        # --------------------------------------------------------

        logits = self.classifier(
            pooled_output
        )

        return logits


# ============================================================================
# Resolve Class Column
# ============================================================================

def _resolve_class_column(
    df: pd.DataFrame,
    path: str,
) -> str:

    """
    Resolve SATD category column.

    Priority:
        1. class
        2. label
        3. column containing SATD category values
    """

    for candidate in ("class", "label"):

        if candidate in df.columns:
            return candidate

    # Fallback
    for col in df.columns:

        values = set(
            df[col]
            .dropna()
            .astype(str)
            .unique()
        )

        # Include original category names
        possible_values = values.intersection(
            set(SATD_CATEGORIES)
            | {
                "code_debt",
                "design_debt",
            }
        )

        if possible_values:

            logger.warning(
                f"Neither 'class' nor 'label' "
                f"found in {path}. "
                f"Using '{col}' as category column."
            )

            return col

    raise ValueError(
        f"Cannot find SATD category column in {path}.\n"
        f"Columns present: {list(df.columns)}"
    )


# ============================================================================
# Load Artifact Data
# ============================================================================

def load_artifact_data(
    artifact_key: str,
    data_dir: str,
    use_augmented: bool = False,
) -> pd.DataFrame:

    """
    Load preprocessed artifact CSV.

    Filters out non-SATD rows.

    Converts:

        code_debt
        design_debt

    into:

        structural_debt
    """

    filename = ARTIFACT_FILES[
        artifact_key
    ]

    path = os.path.join(
        data_dir,
        filename,
    )

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Preprocessed file not found:\n"
            f"{path}\n\n"
            f"Run 01_preprocessing.py first."
        )

    df = pd.read_csv(
        path,
        encoding="utf-8",
    )

    logger.info(
        f"[{artifact_key}] Loaded "
        f"{len(df):,} rows from {path}"
    )

    logger.info(
        f"[{artifact_key}] Columns: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Resolve label column
    # --------------------------------------------------------

    class_col = _resolve_class_column(
        df,
        path,
    )

    if class_col != "class":

        df = df.rename(
            columns={
                class_col: "class"
            }
        )

        logger.info(
            f"[{artifact_key}] "
            f"Renamed '{class_col}' → 'class'"
        )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    df["class"] = (
        df["class"]
        .astype(str)
        .str.strip()
    )

    df["class"] = df["class"].replace(
        {
            "code_debt": "structural_debt",
            "design_debt": "structural_debt",
        }
    )

    # --------------------------------------------------------
    # Keep SATD rows only
    # --------------------------------------------------------

    before = len(df)

    df_satd = df[
        df["class"].isin(
            SATD_CATEGORIES
        )
    ].copy()

    removed = (
        before - len(df_satd)
    )

    logger.info(
        f"[{artifact_key}] SATD rows: "
        f"{len(df_satd):,}"
    )

    logger.info(
        f"[{artifact_key}] Removed: "
        f"{removed:,} non-SATD/unknown rows"
    )

    if len(df_satd) == 0:

        raise ValueError(
            f"[{artifact_key}] No SATD rows found.\n"
            f"Unique classes:\n"
            f"{df['class'].unique().tolist()}\n"
            f"Expected:\n"
            f"{SATD_CATEGORIES}"
        )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    if use_augmented:

        aug_path = os.path.normpath(
            os.path.join(
                data_dir,
                "..",
                f"augmented_{artifact_key.lower()}.csv",
            )
        )

        if os.path.isfile(aug_path):

            df_aug = pd.read_csv(
                aug_path,
                encoding="utf-8",
            )

            aug_col = _resolve_class_column(
                df_aug,
                aug_path,
            )

            if aug_col != "class":

                df_aug = df_aug.rename(
                    columns={
                        aug_col: "class"
                    }
                )

            df_aug["class"] = (
                df_aug["class"]
                .astype(str)
                .str.strip()
            )

            df_aug["class"] = (
                df_aug["class"]
                .replace(
                    {
                        "code_debt":
                            "structural_debt",

                        "design_debt":
                            "structural_debt",
                    }
                )
            )

            df_aug = df_aug[
                df_aug["class"].isin(
                    SATD_CATEGORIES
                )
            ].copy()

            logger.info(
                f"[{artifact_key}] "
                f"Augmented rows: "
                f"{len(df_aug):,}"
            )

            df_satd = pd.concat(
                [
                    df_satd,
                    df_aug,
                ],
                ignore_index=True,
            )

        else:

            logger.warning(
                f"[{artifact_key}] "
                f"Augmentation requested but "
                f"file not found:\n"
                f"{aug_path}"
            )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    logger.info(
        f"[{artifact_key}] "
        f"Final category distribution:"
    )

    distribution = (
        df_satd["class"]
        .value_counts()
    )

    for category in SATD_CATEGORIES:

        count = distribution.get(
            category,
            0,
        )

        logger.info(
            f"    {category:<22} "
            f"{count:>8,}"
        )

    return df_satd


# ============================================================================
# Stratified Split
# ============================================================================

def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
):
    """
    Stratified 80/10/10 split.
    """

    # --------------------------------------------------------
    # Text column
    # --------------------------------------------------------

    if "original_text" in df.columns:

        text_col = "original_text"

    elif "cleaned_text" in df.columns:

        text_col = "cleaned_text"

    else:

        raise ValueError(
            "Neither 'original_text' nor "
            "'cleaned_text' exists.\n"
            f"Columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Remove empty text
    # --------------------------------------------------------

    df = df.copy()

    df[text_col] = (
        df[text_col]
        .fillna("")
        .astype(str)
    )

    df = df[
        df[text_col].str.strip() != ""
    ].copy()

    X = df[text_col].tolist()

    y = df["class"].map(
        LABEL2IDX
    ).tolist()

    # --------------------------------------------------------
    # Check class counts
    # --------------------------------------------------------

    class_counts = (
        pd.Series(y)
        .value_counts()
    )

    logger.info(
        "Class counts before split:"
    )

    for idx, count in class_counts.items():

        logger.info(
            f"    {IDX2LABEL[idx]:<22} "
            f"{count:>8,}"
        )

    if class_counts.min() < 3:

        raise ValueError(
            "Each class needs at least "
            "3 samples for the "
            "80/10/10 split."
        )

    # --------------------------------------------------------
    # 80% train / 20% temporary
    # --------------------------------------------------------

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # --------------------------------------------------------
    # 10% validation / 10% test
    # --------------------------------------------------------

    (
        X_val,
        X_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    logger.info(
        f"Split:"
        f" Train={len(X_train):,}"
        f" | Val={len(X_val):,}"
        f" | Test={len(X_test):,}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        labels = (
            batch["class"]
            .to(device)
        )

        # ----------------------------------------------------
        # token_type_ids optional
        # ----------------------------------------------------

        token_type_ids = batch.get(
            "token_type_ids"
        )

        if token_type_ids is not None:

            token_type_ids = (
                token_type_ids
                .to(device)
            )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * len(labels)
        )

        predictions = (
            logits.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += len(labels)

    average_loss = (
        total_loss / max(total, 1)
    )

    accuracy = (
        correct / max(total, 1)
    )

    return (
        average_loss,
        accuracy,
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["class"]
                .to(device)
            )

            token_type_ids = batch.get(
                "token_type_ids"
            )

            if token_type_ids is not None:

                token_type_ids = (
                    token_type_ids
                    .to(device)
                )

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * len(labels)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            all_predictions.extend(
                predictions
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels
                .cpu()
                .tolist()
            )

    average_loss = (
        total_loss
        / max(len(all_labels), 1)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return (
        average_loss,
        accuracy,
        all_predictions,
        all_labels,
    )


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(
    y_true,
    y_pred,
):

    present_labels = sorted(
        set(y_true)
    )

    target_names = [
        IDX2LABEL[i]
        for i in present_labels
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    return {
        "report": report,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "target_names": target_names,
    }


# ============================================================================
# Result Logging
# ============================================================================

def append_result(
    result_path,
    content,
):

    directory = os.path.dirname(
        result_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        result_path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(content)
        f.write("\n")


# ============================================================================
# Format Result
# ============================================================================

def format_result_block(
    artifact,
    args,
    metrics,
    elapsed,
):

    timestamp = (
        datetime.datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    report = metrics["report"]

    target_names = metrics.get(
        "target_names",
        SATD_CATEGORIES,
    )

    lines = [

        "=" * 80,

        "TRANSFORMER SATD CATEGORIZATION RESULTS",

        f"Timestamp       : {timestamp}",

        f"Artifact        : {artifact}",

        f"Model           : {args.bert_model}",

        "Command Arguments:",

        f"  --artifact       {args.artifact}",
        f"  --mode           {args.mode}",
        f"  --data_dir       {args.data_dir}",
        f"  --bert_model     {args.bert_model}",
        f"  --max_seq_len    {args.max_seq_len}",
        f"  --batch_size     {args.batch_size}",
        f"  --epochs         {args.epochs}",
        f"  --learning_rate  {args.learning_rate}",
        f"  --epsilon        {args.epsilon}",
        f"  --hidden_size    {args.hidden_size}",
        f"  --dropout        {args.dropout}",
        f"  --patience       {args.patience}",
        f"  --use_augmented  {args.use_augmented}",
        f"  --seed           {args.seed}",

        "-" * 80,

        "Test Set Performance:",

        (
            f"  {'Class':<22}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1-Score':>12}"
            f"{'Support':>12}"
        ),

        f"  {'-' * 68}",
    ]

    for cls in target_names:

        if cls in report:

            r = report[cls]

            lines.append(
                f"  {cls:<22}"
                f"{r['precision']:>12.4f}"
                f"{r['recall']:>12.4f}"
                f"{r['f1-score']:>12.4f}"
                f"{int(r['support']):>12}"
            )

    lines += [

        f"  {'-' * 68}",

        (
            f"  {'Macro Avg':<22}"
            f"{report['macro avg']['precision']:>12.4f}"
            f"{report['macro avg']['recall']:>12.4f}"
            f"{metrics['macro_f1']:>12.4f}"
        ),

        (
            f"  {'Weighted F1':<22}"
            f"{metrics['weighted_f1']:>12.4f}"
        ),

        (
            f"  {'Accuracy':<22}"
            f"{metrics['accuracy']:>12.4f}"
        ),

        f"Training time   : {elapsed:.1f}s",

        "=" * 80,

    ]

    return "\n".join(lines) + "\n"


# ============================================================================
# Create Model
# ============================================================================

def create_model(
    model_name,
    num_classes,
    hidden_size,
    dropout,
):

    model = TransformerSATDClassifier(
        model_name=model_name,
        num_classes=num_classes,
        classifier_hidden_size=hidden_size,
        dropout_rate=dropout,
    )

    return model.to(DEVICE)


# ============================================================================
# Train Artifact
# ============================================================================

def run_artifact(
    artifact_key,
    args,
    result_path,
):

    logger.info(
        "\n"
        + "=" * 70
    )

    logger.info(
        f"Transformer Categorization "
        f"— Artifact: {artifact_key}"
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_artifact_data(
        artifact_key,
        args.data_dir,
        args.use_augmented,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = stratified_split(
        df,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    logger.info(
        f"[{artifact_key}] "
        f"Loading tokenizer: "
        f"{args.bert_model}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = SATDCategoryDataset(
        X_train,
        y_train,
        tokenizer,
        args.max_seq_len,
    )

    val_ds = SATDCategoryDataset(
        X_val,
        y_val,
        tokenizer,
        args.max_seq_len,
    )

    test_ds = SATDCategoryDataset(
        X_test,
        y_test,
        tokenizer,
        args.max_seq_len,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_name=args.bert_model,
        num_classes=len(SATD_CATEGORIES),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    unique_classes = np.unique(
        y_train
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # compute_class_weight returns weights only for classes
    # present in y_train.
    #
    # We construct the complete 4-class weight vector.
    # --------------------------------------------------------

    full_class_weights = np.ones(
        len(SATD_CATEGORIES),
        dtype=np.float32,
    )

    for cls, weight in zip(
        unique_classes,
        class_weights,
    ):

        full_class_weights[
            cls
        ] = weight

    class_weights_tensor = torch.tensor(
        full_class_weights,
        dtype=torch.float,
        device=DEVICE,
    )

    logger.info(
        f"[{artifact_key}] "
        f"Class weights: "
        f"{full_class_weights}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=args.label_smoothing,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    total_steps = (
        len(train_loader)
        * args.epochs
    )

    warmup_steps = int(
        total_steps
        * args.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    os.makedirs(
        args.model_dir,
        exist_ok=True,
    )

    model_save_dir = os.path.join(
        args.model_dir,
        f"transformer_{artifact_key}",
    )

    config_path = os.path.join(
        args.model_dir,
        f"transformer_{artifact_key}_config.json",
    )

    best_model_path = os.path.join(
        args.model_dir,
        f"transformer_{artifact_key}_best.pt",
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {

        "model_name":
            args.bert_model,

        "bert_model":
            args.bert_model,

        "max_seq_len":
            args.max_seq_len,

        "num_classes":
            len(SATD_CATEGORIES),

        "hidden_size":
            args.hidden_size,

        "dropout":
            args.dropout,

        "learning_rate":
            args.learning_rate,

        "epsilon":
            args.epsilon,

        "weight_decay":
            args.weight_decay,

        "label_smoothing":
            args.label_smoothing,

        "label2idx":
            LABEL2IDX,

        "idx2label":
            {
                str(k): v
                for k, v in IDX2LABEL.items()
            },

        "artifact":
            artifact_key,

        "categories":
            SATD_CATEGORIES,

        "seed":
            args.seed,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    logger.info(
        f"[{artifact_key}] "
        f"Config saved → "
        f"{config_path}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_val_f1 = -float(
        "inf"
    )

    patience_counter = 0

    start_time = time.time()

    logger.info(
        f"[{artifact_key}] "
        f"Training on {DEVICE}"
    )

    logger.info(
        f"Model       : {args.bert_model}"
    )

    logger.info(
        f"Learning rate: "
        f"{args.learning_rate}"
    )

    logger.info(
        f"Batch size  : "
        f"{args.batch_size}"
    )

    logger.info(
        f"Epochs      : "
        f"{args.epochs}"
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss, train_acc = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                DEVICE,
            )
        )

        (
            val_loss,
            val_acc,
            val_pred,
            val_true,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        val_f1 = f1_score(
            val_true,
            val_pred,
            average="macro",
            zero_division=0,
        )

        logger.info(
            f"[{artifact_key}] "
            f"Epoch "
            f"{epoch:03d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f} | "
            f"TrainAcc: {train_acc:.4f} | "
            f"ValLoss: {val_loss:.4f} | "
            f"ValAcc: {val_acc:.4f} | "
            f"ValMacroF1: {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save based on validation Macro F1
        # ----------------------------------------------------

        improved = (
            val_f1 > best_val_f1
        )

        if improved:

            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            os.makedirs(
                model_save_dir,
                exist_ok=True,
            )

            tokenizer.save_pretrained(
                model_save_dir
            )

            logger.info(
                f"[{artifact_key}] "
                f"Best model saved "
                f"(Val Macro F1="
                f"{best_val_f1:.4f})"
            )

        else:

            patience_counter += 1

            logger.info(
                f"[{artifact_key}] "
                f"No improvement "
                f"({patience_counter}/"
                f"{args.patience})"
            )

            if (
                patience_counter
                >= args.patience
            ):

                logger.info(
                    f"[{artifact_key}] "
                    f"Early stopping at "
                    f"epoch {epoch}"
                )

                break

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    logger.info(
        f"[{artifact_key}] "
        f"Loading best model..."
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    (
        test_loss,
        test_acc,
        y_pred,
        y_true,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    logger.info(
        "\n"
        f"[{artifact_key}] "
        f"TEST RESULTS"
    )

    logger.info(
        f"Macro F1    : "
        f"{metrics['macro_f1']:.4f}"
    )

    logger.info(
        f"Weighted F1 : "
        f"{metrics['weighted_f1']:.4f}"
    )

    logger.info(
        f"Accuracy     : "
        f"{metrics['accuracy']:.4f}"
    )

    logger.info(
        "\n"
        + classification_report(
            y_true,
            y_pred,
            labels=sorted(
                set(y_true)
            ),
            target_names=metrics[
                "target_names"
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_block = format_result_block(
        artifact,
        args,
        metrics,
        elapsed,
    )

    append_result(
        result_path,
        result_block,
    )

    logger.info(
        f"[{artifact_key}] "
        f"Results appended → "
        f"{result_path}"
    )

    return metrics


# ============================================================================
# Evaluate Existing Model
# ============================================================================

def run_evaluate_only(
    artifact_key,
    args,
    result_path,
):

    config_path = os.path.join(
        args.model_dir,
        f"transformer_{artifact_key}_config.json",
    )

    best_model_path = os.path.join(
        args.model_dir,
        f"transformer_{artifact_key}_best.pt",
    )

    model_save_dir = os.path.join(
        args.model_dir,
        f"transformer_{artifact_key}",
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in (
        config_path,
        best_model_path,
    ):

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}\n\n"
                f"Train the model first."
            )

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    model_name = config[
        "model_name"
    ]

    # Backward compatibility
    if not model_name:

        model_name = config[
            "bert_model"
        ]

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer_source = (
        model_save_dir
        if os.path.isdir(
            model_save_dir
        )
        else model_name
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_source
        )
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_artifact_data(
        artifact_key,
        args.data_dir,
        use_augmented=False,
    )

    (
        _,
        _,
        X_test,
        _,
        _,
        y_test,
    ) = stratified_split(
        df,
        seed=config.get(
            "seed",
            42,
        ),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_ds = SATDCategoryDataset(
        X_test,
        y_test,
        tokenizer,
        config[
            "max_seq_len"
        ],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_name=model_name,
        num_classes=config[
            "num_classes"
        ],
        hidden_size=config[
            "hidden_size"
        ],
        dropout=config.get(
            "dropout",
            0.1,
        ),
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Evaluation loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    start = time.time()

    (
        test_loss,
        test_acc,
        y_pred,
        y_true,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
    )

    elapsed = (
        time.time()
        - start
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    logger.info(
        "\n"
        f"[{artifact_key}] "
        f"EVALUATION RESULTS"
    )

    logger.info(
        f"Model       : "
        f"{model_name}"
    )

    logger.info(
        f"Macro F1    : "
        f"{metrics['macro_f1']:.4f}"
    )

    logger.info(
        f"Weighted F1 : "
        f"{metrics['weighted_f1']:.4f}"
    )

    logger.info(
        f"Accuracy     : "
        f"{metrics['accuracy']:.4f}"
    )

    logger.info(
        "\n"
        + classification_report(
            y_true,
            y_pred,
            labels=sorted(
                set(y_true)
            ),
            target_names=metrics[
                "target_names"
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result_block = format_result_block(
        artifact_key,
        args,
        metrics,
        elapsed,
    )

    append_result(
        result_path,
        result_block,
    )

    logger.info(
        f"[{artifact_key}] "
        f"Results appended → "
        f"{result_path}"
    )

    return metrics


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Transformer-based "
            "SATD Categorization"
        ),

        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    parser.add_argument(
        "--artifact",
        type=str,
        default="all",
        choices=[
            "CC",
            "IS",
            "PS",
            "CM",
            "all",
        ],
        help=(
            "Artifact to process. "
            "'all' runs all four."
        ),
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=[
            "train",
            "evaluate",
        ],
        help=(
            "Run mode."
        ),
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            ),
            "data",
            "preprocessed",
        ),
        help=(
            "Directory containing "
            "preprocessed CSV files."
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help=(
            "HuggingFace model identifier. "
            "Examples: "
            "bert-base-uncased, "
            "roberta-base, "
            "microsoft/deberta-v3-base."
        ),
    )

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help=(
            "Maximum token sequence length."
        ),
    )

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=(
            "Training batch size."
        ),
    )

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Maximum number of epochs."
        ),
    )

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help=(
            "AdamW learning rate."
        ),
    )

    # --------------------------------------------------------
    # Epsilon
    # --------------------------------------------------------

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help=(
            "AdamW epsilon."
        ),
    )

    # --------------------------------------------------------
    # Weight decay
    # --------------------------------------------------------

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help=(
            "AdamW weight decay."
        ),
    )

    # --------------------------------------------------------
    # Classifier hidden size
    # --------------------------------------------------------

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help=(
            "Hidden size of "
            "classification head."
        ),
    )

    # --------------------------------------------------------
    # Dropout
    # --------------------------------------------------------

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help=(
            "Classifier dropout."
        ),
    )

    # --------------------------------------------------------
    # Label smoothing
    # --------------------------------------------------------

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help=(
            "CrossEntropy label smoothing."
        ),
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of training steps "
            "used for warmup."
        ),
    )

    # --------------------------------------------------------
    # Patience
    # --------------------------------------------------------

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Early stopping patience."
        ),
    )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    parser.add_argument(
        "--use_augmented",
        action="store_true",
        help=(
            "Include augmented "
            "training samples."
        ),
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed."
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            ),
            "results",
        ),
        help=(
            "Directory for results."
        ),
    )

    # --------------------------------------------------------
    # Model directory
    # --------------------------------------------------------

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            ),
            "models",
        ),
        help=(
            "Directory for model "
            "checkpoints."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    os.makedirs(
        args.model_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Result path
    # --------------------------------------------------------

    result_path = os.path.join(
        args.output_dir,
        "transformer_command_arguments.txt",
    )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    if args.artifact == "all":

        artifacts = list(
            ARTIFACT_FILES.keys()
        )

    else:

        artifacts = [
            args.artifact
        ]

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    all_metrics = {}

    for artifact_key in artifacts:

        try:

            if args.mode == "train":

                metrics = run_artifact(
                    artifact_key,
                    args,
                    result_path,
                )

            else:

                metrics = run_evaluate_only(
                    artifact_key,
                    args,
                    result_path,
                )

            all_metrics[
                artifact_key
            ] = metrics

        except FileNotFoundError as e:

            logger.error(
                str(e)
            )

        except ValueError as e:

            logger.error(
                str(e)
            )

        except Exception as e:

            logger.exception(
                f"[{artifact_key}] "
                f"Unexpected error: "
                f"{e}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if all_metrics:

        logger.info(
            "\n"
            + "=" * 70
        )

        logger.info(
            "TRANSFORMER CATEGORIZATION "
            "SUMMARY"
        )

        logger.info(
            "=" * 70
        )

        logger.info(
            f"Model: "
            f"{args.bert_model}"
        )

        for artifact, metrics in (
            all_metrics.items()
        ):

            logger.info(
                f"  {artifact:4s} → "
                f"Macro F1 = "
                f"{metrics['macro_f1']:.4f} | "
                f"Weighted F1 = "
                f"{metrics['weighted_f1']:.4f} | "
                f"Accuracy = "
                f"{metrics['accuracy']:.4f}"
            )

        logger.info(
            f"\nFull results saved to:\n"
            f"{result_path}"
        )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":

    main()

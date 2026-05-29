"""
04_prediction_pipeline.py
==========================
Two-Step Prediction Pipeline for SATD Detection.

Paper: "Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt"
       Sutoyo et al., 2024 (arXiv:2410.15804v1)

Two-Step Pipeline (paper Section III, Figure 1):
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 1 — IDENTIFICATION (BiLSTM)                            │
    │  Input : Raw text from any artifact (CC, IS, PS, CM)         │
    │  Output: "SATD" or "Not-SATD"                                │
    │          → If Not-SATD: stop here.                           │
    │          → If SATD:    proceed to Step 2.                    │
    └──────────────────────────────────────────────────────────────┘
                              │ SATD only
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 2 — CATEGORIZATION (BERT)                              │
    │  Input : Text flagged as SATD by Step 1                      │
    │  Output: SATD category → C/D | DOC | TES | REQ               │
    └──────────────────────────────────────────────────────────────┘

    Final SATD_label column:
        "Not-SATD"              → not technical debt
        "SATD-C/D"              → code/design debt
        "SATD-DOC"              → documentation debt
        "SATD-TES"              → test debt
        "SATD-REQ"              → requirement debt

Prerequisites:
    - Run 01_preprocessing.py  (builds data/preprocessed/)
    - Run 02_bilstm_identification.py --mode train  (builds models/bilstm_*)
    - Run 03_bert_classification.py   --mode train  (builds models/bert_*)

Usage examples:
    # Predict from a Parquet file (CM artifact, message column):
    python 04_prediction_pipeline.py \\
        --input_file data/commits.parquet \\
        --artifact   CM \\
        --text_col   message

    # Predict from a CSV file (IS artifact):
    python 04_prediction_pipeline.py \\
        --input_file data/new_issues.csv \\
        --artifact   IS \\
        --text_col   text

    # Predict a single text string:
    python 04_prediction_pipeline.py \\
        --text "TODO: This is a hack, needs refactoring later" \\
        --artifact CC

    # All four artifact models:
    python 04_prediction_pipeline.py \\
        --input_file data/mixed.parquet \\
        --artifact   all \\
        --text_col   message

Command-line Arguments:
    --artifact           {CC, IS, PS, CM, all}  Model to use (default: CC)
    --input_file         Path to input CSV or Parquet file
    --text               Single text string to predict
    --text_col           Text column in input file (default: message)
    --output_file        Output file path — .parquet or .csv (default: results/predictions.parquet)
    --model_dir          Directory of saved models (default: models)
    --embedding_dim      GloVe embedding dimension (default: 100)
    --bilstm_threshold   Confidence threshold for SATD (default: 0.5)
    --no_preprocess      Skip text preprocessing

Output:
    Original file with ONE new column added:
        SATD_label : "Not-SATD" | "SATD-C/D" | "SATD-DOC" | "SATD-TES" | "SATD-REQ"
"""

import os
import re
import json
import string
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# Transformers (for BERT step)
try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed. BERT step will be skipped.")

# NLTK (for preprocessing)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not installed. Text preprocessing will be skipped.")

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BILSTM_LABEL2IDX = {"Not-SATD": 0, "SATD": 1}
BILSTM_IDX2LABEL = {0: "Not-SATD", 1: "SATD"}

BERT_CATEGORIES = ["structural_debt", "documentation_debt", "test_debt", "requirement_debt"]
BERT_IDX2LABEL  = {idx: label for idx, label in enumerate(BERT_CATEGORIES)}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

ARTIFACT_KEYS = ["CC", "IS", "PS", "CM"]

# SATD_label format: "Not-SATD" or "SATD-{category}"
NOT_SATD_VALUE = "Not-SATD"

def make_satd_label(bilstm_label: str, bert_category: str) -> str:
    """
    Combine BiLSTM and BERT outputs into a single SATD_label string.
        Not-SATD            → "Not-SATD"
        SATD + C/D          → "SATD-C/D"
        SATD + DOC          → "SATD-DOC"
        SATD + TES          → "SATD-TES"
        SATD + REQ          → "SATD-REQ"
        SATD + no BERT      → "SATD-UNKNOWN"
    """
    if bilstm_label != "SATD":
        return NOT_SATD_VALUE
    if bert_category and bert_category not in ("—", "", None):
        return f"SATD-{bert_category}"
    return "SATD-UNKNOWN"


# ---------------------------------------------------------------------------
# Text Preprocessing (mirrors 01_preprocessing.py exactly)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    if not NLTK_AVAILABLE:
        return False
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}" if res in ("punkt", "punkt_tab") else f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    return True


def preprocess_text(text: str, lemmatizer=None, stop_words=None) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        r"|www\.[^\s]+", " ", text,
    )
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\b\d+\b", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    if not NLTK_AVAILABLE:
        return text
    tokens = word_tokenize(text)
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if len(t) > 2]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# BiLSTM Model (must match 02_bilstm_identification.py exactly)
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=2, pad_idx=0):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm1    = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=True)
        self.dropout1   = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.bilstm2    = nn.LSTM(256, 64,  batch_first=True, bidirectional=True)
        self.dropout2   = nn.Dropout(0.3)
        self.bilstm3    = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.dropout3   = nn.Dropout(0.3)
        self.bilstm4    = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.fc         = nn.Linear(256, num_classes)

    def forward(self, x):
        emb          = self.embedding(x)
        out1, _      = self.bilstm1(emb)
        out1         = self.dropout1(out1)
        last1        = self.batchnorm1(out1[:, -1, :])
        out2, _      = self.bilstm2(out1)
        out2         = self.dropout2(out2)
        out3, _      = self.bilstm3(out2)
        out3         = self.dropout3(out3)
        out4, _      = self.bilstm4(out3)
        return self.fc(out4[:, -1, :])


# ---------------------------------------------------------------------------
# BERT Classifier (must match 03_bert_classification.py exactly)
# ---------------------------------------------------------------------------

class BERTSATDClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=4, hidden_size=256):
        super().__init__()
        self.bert       = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.classifier(
            self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).pooler_output
        )


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class BiLSTMPredictor:
    def __init__(self, model_dir: str, artifact_key: str):
        vocab_path  = os.path.join(model_dir, f"bilstm_{artifact_key}_vocab.json")
        config_path = os.path.join(model_dir, f"bilstm_{artifact_key}_config.json")
        model_path  = os.path.join(model_dir, f"bilstm_{artifact_key}.pt")
        for p in (vocab_path, config_path, model_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"BiLSTM model file not found: {p}\n"
                    f"Run: python 02_bilstm_identification.py --artifact {artifact_key} --mode train"
                )
        with open(vocab_path, encoding="utf-8") as f:
            self.word2idx = json.load(f)
        with open(config_path) as f:
            self.config = json.load(f)
        self.max_seq_len   = self.config["max_seq_len"]
        self.embedding_dim = self.config["embedding_dim"]
        self.pad_idx       = self.config.get("pad_idx", 0)
        self.model = BiLSTMClassifier(
            vocab_size=self.config["vocab_size"],
            embedding_dim=self.embedding_dim,
            num_classes=self.config.get("num_classes", 2),
            pad_idx=self.pad_idx,
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        logger.info(f"BiLSTM [{artifact_key}] loaded.")

    def _encode(self, text: str) -> list:
        tokens  = str(text).split()
        indices = [self.word2idx.get(t, self.word2idx.get(UNK_TOKEN, 1)) for t in tokens]
        if len(indices) >= self.max_seq_len:
            return indices[: self.max_seq_len]
        return indices + [self.pad_idx] * (self.max_seq_len - len(indices))

    @torch.no_grad()
    def predict(self, texts: list, threshold: float = 0.5):
        encoded = [self._encode(t) for t in texts]
        inputs  = torch.tensor(encoded, dtype=torch.long).to(DEVICE)
        probs   = torch.softmax(self.model(inputs), dim=1).cpu().numpy()
        labels, confs = [], []
        for p in probs:
            satd_prob = float(p[BILSTM_LABEL2IDX["SATD"]])
            confs.append(satd_prob)
            labels.append("SATD" if satd_prob >= threshold else "Not-SATD")
        return labels, confs


class BERTPredictor:
    def __init__(self, model_dir: str, artifact_key: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for BERT step.")
        config_path     = os.path.join(model_dir, f"bert_{artifact_key}_config.json")
        best_model_path = os.path.join(model_dir, f"bert_{artifact_key}_best.pt")
        tokenizer_dir   = os.path.join(model_dir, f"bert_{artifact_key}")
        for p in (config_path, best_model_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"BERT model file not found: {p}\n"
                    f"Run: python 03_bert_classification.py --artifact {artifact_key} --mode train"
                )
        with open(config_path) as f:
            self.config = json.load(f)
        self.max_seq_len = self.config["max_seq_len"]
        self.idx2label   = {int(k): v for k, v in self.config["idx2label"].items()}
        tok_src          = tokenizer_dir if os.path.isdir(tokenizer_dir) else self.config["bert_model"]
        self.tokenizer   = BertTokenizer.from_pretrained(tok_src)
        self.model = BERTSATDClassifier(
            bert_model_name=self.config["bert_model"],
            num_classes=self.config["num_classes"],
            hidden_size=self.config["hidden_size"],
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        self.model.eval()
        logger.info(f"BERT [{artifact_key}] loaded.")

    @torch.no_grad()
    def predict(self, texts: list):
        categories, confidences = [], []
        for i in range(0, len(texts), 16):
            batch = texts[i: i + 16]
            enc   = self.tokenizer(
                batch, max_length=self.max_seq_len,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)
            probs = torch.softmax(
                self.model(input_ids, attention_mask, token_type_ids), dim=1
            ).cpu().numpy()
            for p in probs:
                idx = int(np.argmax(p))
                categories.append(self.idx2label[idx])
                confidences.append(float(p[idx]))
        return categories, confidences


# ---------------------------------------------------------------------------
# NLP tools (lazy init)
# ---------------------------------------------------------------------------
_lemmatizer = None
_stop_words = None

def _get_nlp_tools():
    global _lemmatizer, _stop_words
    if _lemmatizer is None and _ensure_nltk():
        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words("english"))
    return _lemmatizer, _stop_words


# ---------------------------------------------------------------------------
# Core pipeline — returns SATD_label list aligned to input texts
# ---------------------------------------------------------------------------

def run_pipeline(
    texts: list,
    artifact_key: str,
    model_dir: str,
    bilstm_threshold: float = 0.5,
    apply_preprocessing: bool = True,
) -> list:
    """
    Run two-step SATD pipeline on a list of texts.

    Returns
    -------
    satd_labels : list of str
        One entry per input text:
            "Not-SATD"   — not technical debt
            "SATD-C/D"   — code/design debt
            "SATD-DOC"   — documentation debt
            "SATD-TES"   — test debt
            "SATD-REQ"   — requirement debt
    """
    logger.info(f"[Pipeline] Artifact: {artifact_key} | Items: {len(texts):,} | Device: {DEVICE}")

    # ── Preprocessing ──────────────────────────────────────────────────────
    if apply_preprocessing:
        lemmatizer, stop_words = _get_nlp_tools()
        preprocessed = [preprocess_text(t, lemmatizer, stop_words) for t in texts]
    else:
        preprocessed = [str(t) for t in texts]

    # ── Step 1: BiLSTM identification ──────────────────────────────────────
    logger.info("[Step 1] Running BiLSTM identification …")
    bilstm        = BiLSTMPredictor(model_dir, artifact_key)
    bilstm_labels, _ = bilstm.predict(preprocessed, threshold=bilstm_threshold)

    satd_indices = [i for i, lbl in enumerate(bilstm_labels) if lbl == "SATD"]
    logger.info(
        f"[Step 1] SATD: {len(satd_indices):,} | "
        f"Not-SATD: {len(texts) - len(satd_indices):,}"
    )

    # ── Step 2: BERT categorization (SATD items only) ──────────────────────
    bert_categories = [""] * len(texts)

    if satd_indices and TRANSFORMERS_AVAILABLE:
        logger.info(f"[Step 2] Running BERT categorization on {len(satd_indices):,} SATD items …")
        try:
            bert       = BERTPredictor(model_dir, artifact_key)
            # Use ORIGINAL text for BERT (it has its own tokeniser)
            satd_texts = [texts[i] for i in satd_indices]
            cats, _    = bert.predict(satd_texts)
            for j, idx in enumerate(satd_indices):
                bert_categories[idx] = cats[j]
            logger.info("[Step 2] BERT categorization complete.")
        except FileNotFoundError as e:
            logger.warning(f"[Step 2] BERT model not found — SATD items will be 'SATD-UNKNOWN': {e}")
    elif not TRANSFORMERS_AVAILABLE:
        logger.warning("[Step 2] transformers not available. SATD items labelled 'SATD-UNKNOWN'.")
    else:
        logger.info("[Step 2] No SATD items — BERT step skipped.")

    # ── Build final SATD_label column ──────────────────────────────────────
    satd_labels = [
        make_satd_label(bilstm_labels[i], bert_categories[i])
        for i in range(len(texts))
    ]

    # Summary
    from collections import Counter
    dist = Counter(satd_labels)
    logger.info("[Pipeline] SATD_label distribution:")
    for lbl, cnt in sorted(dist.items()):
        logger.info(f"  {lbl:<20}: {cnt:,}")

    return satd_labels


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_input_file(path: str) -> pd.DataFrame:
    """Load CSV or Parquet, preserving all original columns."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
        logger.info(f"Loaded Parquet: {path} ({len(df):,} rows, {len(df.columns)} cols)")
    elif ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        logger.info(f"Loaded CSV: {path} ({len(df):,} rows, {len(df.columns)} cols)")
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Use .parquet or .csv")
    return df


def save_output_file(df: pd.DataFrame, path: str):
    """Save DataFrame to Parquet or CSV based on output path extension."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        # Default to parquet if extension is unrecognised
        path = path + ".parquet"
        df.to_parquet(path, index=False)
    logger.info(f"Output saved → {path}  ({len(df):,} rows, columns: {list(df.columns)})")


def resolve_text_column(df: pd.DataFrame, requested_col: str) -> str:
    """
    Return the text column to use.  Priority:
      1. Exactly the column the user requested (--text_col)
      2. Common fallback names
    """
    if requested_col in df.columns:
        return requested_col
    fallbacks = ["message", "text", "comment", "body", "content",
                 "commit_message", "summary", "description"]
    for col in fallbacks:
        if col in df.columns:
            logger.warning(
                f"Column '{requested_col}' not found. Using '{col}' instead."
            )
            return col
    raise ValueError(
        f"Cannot find text column '{requested_col}' in file.\n"
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Step SATD Prediction Pipeline (Sutoyo et al., 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact", type=str, default="CM",
        choices=["CC", "IS", "PS", "CM", "all"],
        help="Artifact model(s) to use. 'all' runs all four.",
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to input .parquet or .csv file.",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Single raw text string to predict (alternative to --input_file).",
    )
    parser.add_argument(
        "--text_col", type=str, default="message",
        help="Column in --input_file that contains the text to classify.",
    )
    parser.add_argument(
        "--output_file", type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "predictions.parquet"
        ),
        help="Output file path (.parquet or .csv). All original columns are preserved.",
    )
    parser.add_argument(
        "--model_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Directory containing trained BiLSTM and BERT model files.",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=100,
        help="GloVe embedding dimension (must match training).",
    )
    parser.add_argument(
        "--bilstm_threshold", type=float, default=0.5,
        help="Minimum SATD probability for BiLSTM to label a text as SATD.",
    )
    parser.add_argument(
        "--no_preprocess", action="store_true",
        help="Skip text preprocessing (use raw text as-is).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    apply_preprocessing = not args.no_preprocess

    # ── Single text mode ──────────────────────────────────────────────────
    if args.text:
        texts  = [args.text]
        labels = run_pipeline(
            texts, args.artifact, args.model_dir,
            args.bilstm_threshold, apply_preprocessing,
        )
        print(f"\nInput : {args.text}")
        print(f"Result: {labels[0]}\n")
        return

    # ── File mode ─────────────────────────────────────────────────────────
    if not args.input_file:
        # Demo mode
        logger.info("No --input_file or --text provided. Running demo …")
        demo = [
            "TODO: this is a hack, needs to be refactored properly later",
            "Fix the login bug introduced in last commit",
            "FIXME: temporary workaround for the authentication issue",
            "Add unit tests for the payment module",
            "Update the README with installation steps",
        ]
        labels = run_pipeline(
            demo, args.artifact, args.model_dir,
            args.bilstm_threshold, apply_preprocessing,
        )
        for text, lbl in zip(demo, labels):
            print(f"  [{lbl:<18}]  {text}")
        return

    if not os.path.isfile(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return

    df       = load_input_file(args.input_file)
    text_col = resolve_text_column(df, args.text_col)
    texts    = df[text_col].fillna("").astype(str).tolist()

    if args.artifact == "all":
        # Run all four models; add one SATD_label column per artifact
        for artifact_key in ARTIFACT_KEYS:
            col_name = f"SATD_label_{artifact_key}"
            try:
                labels = run_pipeline(
                    texts, artifact_key, args.model_dir,
                    args.bilstm_threshold, apply_preprocessing,
                )
                df[col_name] = labels
                logger.info(f"Added column: {col_name}")
            except FileNotFoundError as e:
                logger.warning(f"Skipping {artifact_key}: {e}")
    else:
        # Single artifact → one column named SATD_label
        labels = run_pipeline(
            texts, args.artifact, args.model_dir,
            args.bilstm_threshold, apply_preprocessing,
        )
        df["SATD_label"] = labels

    save_output_file(df, args.output_file)
    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# Public API for programmatic use
# ---------------------------------------------------------------------------

def predict(
    texts,
    artifact: str = "CM",
    model_dir: str = None,
    bilstm_threshold: float = 0.5,
    preprocess: bool = True,
) -> list:
    """
    Programmatic API — returns a list of SATD_label strings.

    Parameters
    ----------
    texts            : str or list of str
    artifact         : 'CC', 'IS', 'PS', or 'CM'
    model_dir        : path to models/ directory
    bilstm_threshold : BiLSTM confidence cutoff (default 0.5)
    preprocess       : apply text preprocessing (default True)

    Returns
    -------
    list of str: "Not-SATD" | "SATD-C/D" | "SATD-DOC" | "SATD-TES" | "SATD-REQ"

    Example
    -------
    >>> from prediction_pipeline import predict
    >>> predict("TODO: fix this ugly hack", artifact="CM")
    ['SATD-C/D']
    """
    if isinstance(texts, str):
        texts = [texts]
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return run_pipeline(texts, artifact, model_dir, bilstm_threshold, preprocess)


def predict_dataframe(
    df: pd.DataFrame,
    text_col: str = "message",
    artifact: str = "CM",
    model_dir: str = None,
    bilstm_threshold: float = 0.5,
    preprocess: bool = True,
    label_col: str = "SATD_label",
) -> pd.DataFrame:
    """
    Convenience wrapper: takes a DataFrame, adds SATD_label column, returns it.

    Example
    -------
    >>> import pandas as pd
    >>> from prediction_pipeline import predict_dataframe
    >>> df = pd.read_parquet("commits.parquet")
    >>> df = predict_dataframe(df, text_col="message", artifact="CM")
    >>> df.to_parquet("commits_with_satd.parquet", index=False)
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = run_pipeline(texts, artifact, model_dir, bilstm_threshold, preprocess)
    df = df.copy()
    df[label_col] = labels
    return df


if __name__ == "__main__":
    main()

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
    │  Output: SATD category → C/D | DOC | TES | REQ              │
    └──────────────────────────────────────────────────────────────┘

Prerequisites:
    - Run 01_preprocessing.py  (builds data/preprocessed/)
    - Run 02_bilstm_identification.py --mode train  (builds models/bilstm_*)
    - Run 03_bert_classification.py   --mode train  (builds models/bert_*)

Usage examples:
    # Predict from a CSV file (IS artifact):
    python 04_prediction_pipeline.py \\
        --input_csv  data/new_issues.csv \\
        --artifact   IS \\
        --text_col   text

    # Predict a single text string (CC artifact, default):
    python 04_prediction_pipeline.py \\
        --text "TODO: This is a hack, needs refactoring later" \\
        --artifact CC

    # Predict from CSV and save results to a custom output file:
    python 04_prediction_pipeline.py \\
        --input_csv  data/commits.csv \\
        --artifact   CM \\
        --text_col   message \\
        --output_csv results/predictions.csv

    # Use all four artifact models in sequence (ensemble / multi-artifact):
    python 04_prediction_pipeline.py \\
        --input_csv  data/mixed.csv \\
        --artifact   all \\
        --text_col   text

Command-line Arguments:
    --artifact      {CC, IS, PS, CM, all}  Artifact model to use (default: CC)
    --input_csv     Path to input CSV file (optional; use --text for single input)
    --text          Single text string to predict (optional)
    --text_col      Column name for text in CSV (default: text)
    --output_csv    Path to save predictions CSV (default: results/predictions.csv)
    --model_dir     Directory of saved models (default: models)
    --glove_path    Path to GloVe embeddings (used only if embedding cache missing)
    --embedding_dim GloVe dimension (default: 100)
    --bilstm_threshold  Confidence threshold for SATD (default: 0.5)
    --no_preprocess Flag to skip text preprocessing (use raw text as-is)

Output:
    - results/predictions.csv  : CSV with columns:
          text, bilstm_label, bilstm_confidence, bert_category, bert_confidence
    - Console print of per-item results
"""

import os
import re
import json
import string
import logging
import argparse
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

BERT_CATEGORIES = ["C/D", "DOC", "TES", "REQ"]
BERT_LABEL2IDX  = {label: idx for idx, label in enumerate(BERT_CATEGORIES)}
BERT_IDX2LABEL  = {idx: label for label, idx in BERT_LABEL2IDX.items()}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

ARTIFACT_KEYS = ["CC", "IS", "PS", "CM"]

# ---------------------------------------------------------------------------
# Text Preprocessing (mirrors 01_preprocessing.py)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    if not NLTK_AVAILABLE:
        return False
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for res in resources:
        try:
            if res in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{res}")
            else:
                nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    return True


def preprocess_text(text: str, lemmatizer=None, stop_words=None) -> str:
    """
    Identical preprocessing pipeline from 01_preprocessing.py.
    Applied to raw input before feeding to BiLSTM.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        r"|www\.[^\s]+",
        " ", text,
    )

    # 3. Remove non-ASCII
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 4. Remove standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # 5. Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not NLTK_AVAILABLE:
        return text

    # 7. Tokenize
    tokens = word_tokenize(text)

    # 8. Stop word removal
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]

    # 9. Remove short words (≤ 2 chars)
    tokens = [t for t in tokens if len(t) > 2]

    # 10. Lemmatize
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# BiLSTM Model (must match 02_bilstm_identification.py exactly)
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """
    Stacked Bidirectional LSTM — must match architecture in 02_bilstm_identification.py.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int = 2,
        pad_idx: int = 0,
    ):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.bilstm1   = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=True)
        self.dropout1  = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.bilstm2   = nn.LSTM(256, 64,  batch_first=True, bidirectional=True)
        self.dropout2  = nn.Dropout(0.3)

        self.bilstm3   = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.dropout3  = nn.Dropout(0.3)

        self.bilstm4   = nn.LSTM(256, 128, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        emb  = self.embedding(x)
        out1, _ = self.bilstm1(emb)
        out1 = self.dropout1(out1)
        last1 = out1[:, -1, :]
        last1 = self.batchnorm1(last1)
        out2, _ = self.bilstm2(out1)
        out2 = self.dropout2(out2)
        out3, _ = self.bilstm3(out2)
        out3 = self.dropout3(out3)
        out4, _ = self.bilstm4(out3)
        last4 = out4[:, -1, :]
        return self.fc(last4)


# ---------------------------------------------------------------------------
# BERT Classifier (must match 03_bert_classification.py exactly)
# ---------------------------------------------------------------------------

class BERTSATDClassifier(nn.Module):
    """
    BERT-base-uncased classifier — must match architecture in 03_bert_classification.py.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        hidden_size: int = 256,
    ):
        super(BERTSATDClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(outputs.pooler_output)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

class BiLSTMPredictor:
    """Wraps a trained BiLSTM model for single/batch inference."""

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

        self.max_seq_len = self.config["max_seq_len"]
        self.embedding_dim = self.config["embedding_dim"]
        self.pad_idx = self.config.get("pad_idx", 0)

        self.model = BiLSTMClassifier(
            vocab_size=self.config["vocab_size"],
            embedding_dim=self.embedding_dim,
            num_classes=self.config.get("num_classes", 2),
            pad_idx=self.pad_idx,
        ).to(DEVICE)

        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        self.model.eval()
        logger.info(f"BiLSTM [{artifact_key}] loaded from {model_path}")

    def _encode(self, text: str) -> list:
        tokens = str(text).split()
        indices = [self.word2idx.get(t, self.word2idx.get(UNK_TOKEN, 1)) for t in tokens]
        if len(indices) >= self.max_seq_len:
            return indices[: self.max_seq_len]
        return indices + [self.pad_idx] * (self.max_seq_len - len(indices))

    @torch.no_grad()
    def predict(self, texts: list, threshold: float = 0.5):
        """
        Predict SATD vs Not-SATD for a list of (preprocessed) texts.

        Returns
        -------
        labels       : list of str  → 'SATD' or 'Not-SATD'
        confidences  : list of float → probability of 'SATD'
        """
        encoded = [self._encode(t) for t in texts]
        inputs  = torch.tensor(encoded, dtype=torch.long).to(DEVICE)
        logits  = self.model(inputs)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()

        labels = []
        confidences = []
        for p in probs:
            satd_prob = float(p[BILSTM_LABEL2IDX["SATD"]])
            confidences.append(satd_prob)
            labels.append("SATD" if satd_prob >= threshold else "Not-SATD")
        return labels, confidences


class BERTPredictor:
    """Wraps a trained BERT model for single/batch inference."""

    def __init__(self, model_dir: str, artifact_key: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for BERT step. "
                "Install with: pip install transformers"
            )

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

        # Load tokenizer
        tok_source = tokenizer_dir if os.path.isdir(tokenizer_dir) else self.config["bert_model"]
        self.tokenizer = BertTokenizer.from_pretrained(tok_source)

        # Load model
        self.model = BERTSATDClassifier(
            bert_model_name=self.config["bert_model"],
            num_classes=self.config["num_classes"],
            hidden_size=self.config["hidden_size"],
        ).to(DEVICE)
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=DEVICE)
        )
        self.model.eval()
        logger.info(f"BERT [{artifact_key}] loaded from {best_model_path}")

    @torch.no_grad()
    def predict(self, texts: list):
        """
        Categorize SATD types for a list of texts.

        Returns
        -------
        categories   : list of str  → 'C/D', 'DOC', 'TES', or 'REQ'
        confidences  : list of float → probability of predicted category
        """
        categories  = []
        confidences = []

        # Process in small batches to avoid OOM
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)
            token_type_ids = encodings.get(
                "token_type_ids",
                torch.zeros_like(input_ids),
            ).to(DEVICE)

            logits = self.model(input_ids, attention_mask, token_type_ids)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

            for p in probs:
                pred_idx = int(np.argmax(p))
                categories.append(self.idx2label[pred_idx])
                confidences.append(float(p[pred_idx]))

        return categories, confidences


# ---------------------------------------------------------------------------
# Preprocessing state (lazy init)
# ---------------------------------------------------------------------------
_lemmatizer = None
_stop_words = None


def _get_nlp_tools():
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        if _ensure_nltk():
            _lemmatizer = WordNetLemmatizer()
            _stop_words = set(stopwords.words("english"))
    return _lemmatizer, _stop_words


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    texts: list,
    artifact_key: str,
    model_dir: str,
    bilstm_threshold: float = 0.5,
    apply_preprocessing: bool = True,
) -> pd.DataFrame:
    """
    Execute the two-step SATD prediction pipeline.

    Step 1 — BiLSTM identifies SATD vs Not-SATD.
    Step 2 — BERT categorizes identified SATD into C/D, DOC, TES, REQ.

    Parameters
    ----------
    texts               : List of raw text strings.
    artifact_key        : One of 'CC', 'IS', 'PS', 'CM'.
    model_dir           : Directory containing saved model files.
    bilstm_threshold    : Probability threshold for SATD decision (default 0.5).
    apply_preprocessing : Whether to preprocess texts before BiLSTM.

    Returns
    -------
    DataFrame with columns:
        original_text, preprocessed_text,
        bilstm_label, bilstm_confidence,
        bert_category, bert_confidence
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Two-Step SATD Pipeline — Artifact: {artifact_key}")
    logger.info(f"  Items to process : {len(texts)}")
    logger.info(f"  Device           : {DEVICE}")
    logger.info(f"  BiLSTM threshold : {bilstm_threshold}")
    logger.info(f"{'='*60}")

    # ── Preprocessing ──────────────────────────────────────────────────────
    if apply_preprocessing:
        lemmatizer, stop_words = _get_nlp_tools()
        preprocessed = [preprocess_text(t, lemmatizer, stop_words) for t in texts]
    else:
        preprocessed = [str(t) for t in texts]

    # ── Step 1: BiLSTM identification ──────────────────────────────────────
    logger.info(f"[Step 1] Loading BiLSTM model for artifact: {artifact_key}")
    bilstm = BiLSTMPredictor(model_dir, artifact_key)
    bilstm_labels, bilstm_confs = bilstm.predict(preprocessed, threshold=bilstm_threshold)

    satd_mask    = [label == "SATD" for label in bilstm_labels]
    satd_indices = [i for i, m in enumerate(satd_mask) if m]
    logger.info(
        f"[Step 1] Results — SATD: {sum(satd_mask)} | "
        f"Not-SATD: {len(texts) - sum(satd_mask)}"
    )

    # ── Step 2: BERT categorization (only for SATD items) ──────────────────
    bert_categories = ["—"] * len(texts)   # default: not applicable
    bert_confs      = [0.0]  * len(texts)

    if satd_indices and TRANSFORMERS_AVAILABLE:
        logger.info(
            f"[Step 2] Loading BERT model for {len(satd_indices)} SATD items …"
        )
        try:
            bert = BERTPredictor(model_dir, artifact_key)
            satd_texts = [texts[i] for i in satd_indices]  # use original text for BERT
            cats, confs = bert.predict(satd_texts)
            for j, idx in enumerate(satd_indices):
                bert_categories[idx] = cats[j]
                bert_confs[idx]      = confs[j]
            logger.info(f"[Step 2] BERT categorization complete.")
        except FileNotFoundError as e:
            logger.warning(f"[Step 2] BERT model not found, skipping: {e}")
    elif not TRANSFORMERS_AVAILABLE:
        logger.warning("[Step 2] transformers not available. Skipping BERT categorization.")
    else:
        logger.info("[Step 2] No SATD items identified — BERT step skipped.")

    # ── Assemble results DataFrame ─────────────────────────────────────────
    results = pd.DataFrame({
        "original_text":     texts,
        "preprocessed_text": preprocessed,
        "bilstm_label":      bilstm_labels,
        "bilstm_confidence": [round(c, 4) for c in bilstm_confs],
        "bert_category":     bert_categories,
        "bert_confidence":   [round(c, 4) for c in bert_confs],
    })

    return results


def run_multi_artifact_pipeline(
    texts: list,
    model_dir: str,
    bilstm_threshold: float = 0.5,
    apply_preprocessing: bool = True,
) -> dict:
    """
    Run the pipeline with ALL four artifact models and return results for each.
    Useful when the artifact type of the input is unknown.
    """
    all_results = {}
    for artifact_key in ARTIFACT_KEYS:
        try:
            df = run_pipeline(
                texts, artifact_key, model_dir, bilstm_threshold, apply_preprocessing
            )
            all_results[artifact_key] = df
        except FileNotFoundError as e:
            logger.warning(f"Skipping artifact {artifact_key}: {e}")
    return all_results


# ---------------------------------------------------------------------------
# Result display & saving
# ---------------------------------------------------------------------------

def print_results(results: pd.DataFrame, artifact_key: str):
    """Pretty-print prediction results to console."""
    print(f"\n{'='*70}")
    print(f"Two-Step SATD Prediction Results  |  Artifact: {artifact_key}")
    print(f"{'='*70}")
    print(
        f"{'#':<5} {'BiLSTM':>10} {'Conf':>7} {'BERT Cat':>10} {'Conf':>7}  Text"
    )
    print(f"{'-'*70}")
    for i, row in results.iterrows():
        text_preview = str(row["original_text"])[:50].replace("\n", " ")
        bert_cat = str(row["bert_category"]) if row["bilstm_label"] == "SATD" else "—"
        bert_conf = f"{row['bert_confidence']:.3f}" if row["bilstm_label"] == "SATD" else "  —  "
        print(
            f"{i+1:<5} {row['bilstm_label']:>10} {row['bilstm_confidence']:>7.3f} "
            f"{bert_cat:>10} {bert_conf:>7}  {text_preview}"
        )

    satd_count = (results["bilstm_label"] == "SATD").sum()
    total      = len(results)
    print(f"\n  Total: {total} | SATD: {satd_count} | Not-SATD: {total - satd_count}")

    if satd_count > 0:
        print("\n  SATD Category Breakdown:")
        cat_counts = (
            results[results["bilstm_label"] == "SATD"]["bert_category"]
            .value_counts()
        )
        for cat, cnt in cat_counts.items():
            print(f"    {cat:<8}: {cnt}")
    print(f"{'='*70}\n")


def save_results(results: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Predictions saved → {output_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Step SATD Prediction Pipeline (Sutoyo et al., 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="CC",
        choices=["CC", "IS", "PS", "CM", "all"],
        help=(
            "Artifact model(s) to use for prediction. "
            "'all' runs all four models independently."
        ),
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to a CSV file containing texts to predict.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single raw text string to predict (alternative to --input_csv).",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Column name in --input_csv that contains the text.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "predictions.csv"
        ),
        help="Path to save prediction results CSV.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Directory containing trained model files.",
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "glove.6B.100d.txt"
        ),
        help="Path to GloVe embeddings (not used at inference, kept for reference).",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="GloVe embedding dimension (must match training).",
    )
    parser.add_argument(
        "--bilstm_threshold",
        type=float,
        default=0.5,
        help="Minimum SATD probability for BiLSTM to classify as SATD.",
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="Skip text preprocessing (use raw input text as-is).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Collect input texts ───────────────────────────────────────────────
    texts = []

    if args.text:
        texts = [args.text]
        logger.info(f"Single text input: '{args.text[:80]}'")

    elif args.input_csv:
        if not os.path.isfile(args.input_csv):
            logger.error(f"Input CSV not found: {args.input_csv}")
            return
        df_input = pd.read_csv(args.input_csv, encoding="utf-8", on_bad_lines="skip")
        if args.text_col not in df_input.columns:
            # Try fallback column names
            fallbacks = [
                "text", "comment", "message", "body", "content",
                "commit_message", "summary", "description",
            ]
            found_col = next((c for c in fallbacks if c in df_input.columns), None)
            if found_col:
                logger.warning(
                    f"Column '{args.text_col}' not found. Using '{found_col}' instead."
                )
                args.text_col = found_col
            else:
                logger.error(
                    f"Text column '{args.text_col}' not found in {args.input_csv}. "
                    f"Available: {list(df_input.columns)}"
                )
                return
        texts = df_input[args.text_col].fillna("").tolist()
        logger.info(f"Loaded {len(texts):,} texts from {args.input_csv}")

    else:
        # Interactive demo mode
        logger.info("No --text or --input_csv provided. Running interactive demo …")
        demo_texts = [
            "TODO: this is a hack, needs to be refactored properly later",
            "Fix the login bug introduced in last commit",
            "FIXME: temporary workaround for the authentication issue",
            "Add unit tests for the payment module",
            "Update the README with installation steps",
            "This implementation is not optimal but it works for now",
        ]
        logger.info("Demo texts:")
        for i, t in enumerate(demo_texts, 1):
            logger.info(f"  {i}. {t}")
        texts = demo_texts

    if not texts:
        logger.error("No texts to process.")
        return

    # ── Run pipeline ──────────────────────────────────────────────────────
    apply_preprocessing = not args.no_preprocess

    if args.artifact == "all":
        all_results = run_multi_artifact_pipeline(
            texts,
            model_dir=args.model_dir,
            bilstm_threshold=args.bilstm_threshold,
            apply_preprocessing=apply_preprocessing,
        )
        if not all_results:
            logger.error("No models found. Train models first.")
            return

        # Print results for each artifact model
        for artifact_key, results in all_results.items():
            print_results(results, artifact_key)

        # Save the last artifact results as CSV (or combine them)
        last_key  = list(all_results.keys())[-1]
        last_df   = all_results[last_key]
        save_results(last_df, args.output_csv)

    else:
        try:
            results = run_pipeline(
                texts,
                artifact_key=args.artifact,
                model_dir=args.model_dir,
                bilstm_threshold=args.bilstm_threshold,
                apply_preprocessing=apply_preprocessing,
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        print_results(results, args.artifact)
        save_results(results, args.output_csv)

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# Public API for programmatic use
# ---------------------------------------------------------------------------

def predict(
    texts,
    artifact: str = "CC",
    model_dir: str = None,
    bilstm_threshold: float = 0.5,
    preprocess: bool = True,
) -> pd.DataFrame:
    """
    Convenience function for programmatic use of the pipeline.

    Parameters
    ----------
    texts            : str or list of str — text(s) to classify.
    artifact         : Artifact type ('CC', 'IS', 'PS', 'CM').
    model_dir        : Path to trained models directory.
    bilstm_threshold : SATD confidence threshold for BiLSTM.
    preprocess       : Whether to apply text preprocessing.

    Returns
    -------
    pd.DataFrame with prediction results.

    Example
    -------
    >>> from 04_prediction_pipeline import predict
    >>> df = predict("TODO: fix this ugly hack", artifact="CC")
    >>> print(df[["bilstm_label", "bert_category"]])
    """
    if isinstance(texts, str):
        texts = [texts]

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    return run_pipeline(
        texts=texts,
        artifact_key=artifact,
        model_dir=model_dir,
        bilstm_threshold=bilstm_threshold,
        apply_preprocessing=preprocess,
    )


if __name__ == "__main__":
    main()
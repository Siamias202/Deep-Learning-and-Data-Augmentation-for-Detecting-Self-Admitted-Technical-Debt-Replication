"""
01_preprocessing.py
====================
Text Preprocessing for Self-Admitted Technical Debt (SATD) Detection.

Paper: "Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Debt"
       Sutoyo et al., 2024 (arXiv:2410.15804v1)

Paper methodology (Section III-D):
    Standard preprocessing procedures:
    - Data cleansing
    - Duplicate removal
    - Lowercase conversion
    - Tokenization
    - Stop word removal
    - Punctuation removal
    - Lemmatization
    - Excluding short words (2 letters or fewer)
    - Removing numbers, URLs, and non-ASCII characters
    - Eliminating extra white spaces

Data:
    Four artifact CSV files:
      - data-augmentation-code_comments.csv
      - data-augmentation-commit-messages.csv
      - data-augmentation-issues.csv
      - data-augmentation-pull-requests.csv

    Expected columns: text (or comment/message), label (SATD type or Not-SATD)

Output:
    Preprocessed CSV files saved to data/preprocessed/ directory.
    Each file contains:
      - original_text: raw text
      - cleaned_text:  preprocessed text
      - label:         original label
      - binary_label:  'SATD' or 'Not-SATD' (for BiLSTM identification step)
"""

import os
import re
import string
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
# NLTK resource download
# ---------------------------------------------------------------------------
NLTK_RESOURCES = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]

def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    for resource in NLTK_RESOURCES:
        try:
            if resource in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{resource}")
            elif resource in ("stopwords",):
                nltk.data.find(f"corpora/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Artifact file configuration
# ---------------------------------------------------------------------------
ARTIFACT_FILES = {
    "CC": "data-augmentation-code_comments.csv",
    "CM": "data-augmentation-commit-messages.csv",
    "IS": "data-augmentation-issues.csv",
    "PS": "data-augmentation-pull-requests.csv",
}

# SATD type labels as per Li et al. [20] and the paper (Table I)
SATD_TYPES = {"code_debt", "design_debt", "test_debt", "requirement_debt", "documentation_debt"} # for commit messages , issues, pull requests
NOT_SATD_LABEL = "non_debt"


# Possible column name variants for the raw text field across different CSVs
TEXT_COLUMN_CANDIDATES = [
    "text", "comment", "message", "body", "content",
    "commit_message", "issue_text", "pull_request_text",
    "code_comment", "summary", "description",
]

# Possible column name variants for the label field
LABEL_COLUMN_CANDIDATES = [
    "label", "class", "type", "satd_type", "category", "classification",
]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www. links."""
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        r"|www\.[^\s]+"
    )
    return url_pattern.sub(" ", text)


def _remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters (e.g., emojis, special Unicode)."""
    return text.encode("ascii", errors="ignore").decode("ascii")


def _remove_numbers(text: str) -> str:
    """Remove standalone numbers; keeps alphanumeric tokens like 'v2' intact."""
    return re.sub(r"\b\d+\b", " ", text)


def _remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def _remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_short_words(tokens: list, min_length: int = 3) -> list:
    """
    Remove words with length <= 2 characters (paper: 'excluding short words
    two letters or fewer').
    """
    return [t for t in tokens if len(t) > min_length - 1]


def preprocess_text(
    text: str,
    lemmatizer: WordNetLemmatizer,
    stop_words: set,
) -> str:
    """
    Full preprocessing pipeline as described in Section III-D of the paper.

    Steps (in order):
    1.  Lowercase conversion
    2.  Remove URLs
    3.  Remove non-ASCII characters
    4.  Remove numbers
    5.  Remove punctuation
    6.  Remove extra whitespace
    7.  Tokenization
    8.  Stop word removal
    9.  Remove short words (≤ 2 chars)
    10. Lemmatization

    Parameters
    ----------
    text       : Raw input string.
    lemmatizer : Shared WordNetLemmatizer instance.
    stop_words : Set of stopwords to remove.

    Returns
    -------
    Preprocessed string (space-joined tokens).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = _remove_urls(text)

    # 3. Remove non-ASCII
    text = _remove_non_ascii(text)

    # 4. Remove numbers
    text = _remove_numbers(text)

    # 5. Remove punctuation
    text = _remove_punctuation(text)

    # 6. Remove extra whitespace
    text = _remove_extra_whitespace(text)

    # 7. Tokenize
    tokens = word_tokenize(text)

    # 8. Stop word removal
    tokens = [t for t in tokens if t not in stop_words]

    # 9. Remove short words (paper: exclude words ≤ 2 letters)
    tokens = _remove_short_words(tokens, min_length=3)

    # 10. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def resolve_text_column(df: pd.DataFrame) -> str:
    """Return the first matching text column name found in the DataFrame."""
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    # Fall back: first string-dtype column
    str_cols = [c for c in df.columns if df[c].dtype == object]
    if str_cols:
        logger.warning(
            f"No standard text column found. Using first object column: '{str_cols[0]}'"
        )
        return str_cols[0]
    raise ValueError("Cannot identify text column in the DataFrame.")


def resolve_label_column(df: pd.DataFrame) -> str:
    """Return the first matching label column name found in the DataFrame."""
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Cannot identify label column. Expected one of: {LABEL_COLUMN_CANDIDATES}. "
        f"Found columns: {list(df.columns)}"
    )


def make_binary_label(label: str) -> str:
    """
    Convert multi-class SATD type label to binary label for BiLSTM step.

    Per paper Section III-B:
        'we merge all types of SATD (C/D, DOC, TES, and REQ) into one class,
         namely "SATD", so that there are only two classes (SATD and Not-SATD).'
    """
    label_str = str(label).strip()
    if label_str == NOT_SATD_LABEL:
        return NOT_SATD_LABEL
    # Any SATD type maps to 'SATD'
    return "SATD"

def categorize_label(label: str) -> str:
    """
    Map multi-class SATD type label to a standardized category.

    Per paper Table I, the SATD types are:
        - C/D: Code/Design Debt
        - DOC: Documentation Debt
        - TES: Test Debt
        - REQ: Requirement Debt
        - SATD: General SATD (unspecified type)
    This function can be used to ensure consistent labeling across datasets.
    """
    label_str = str(label).strip()
    if label_str in SATD_TYPES:
        return label_str
    # If it's not a recognized SATD type, treat it as Not-SATD
    return NOT_SATD_LABEL


# ---------------------------------------------------------------------------
# Per-artifact processing
# ---------------------------------------------------------------------------

def process_artifact(
    artifact_key: str,
    file_path: str,
    output_dir: str,
    lemmatizer: WordNetLemmatizer,
    stop_words: set,
) -> pd.DataFrame:
    """
    Load, preprocess, and save one artifact CSV file.

    Parameters
    ----------
    artifact_key : Short key, e.g. 'CC', 'CM', 'IS', 'PS'.
    file_path    : Full path to the raw CSV.
    output_dir   : Directory where the preprocessed CSV will be saved.
    lemmatizer   : Shared WordNetLemmatizer instance.
    stop_words   : Set of English stop words.

    Returns
    -------
    Preprocessed DataFrame (also written to disk).
    """
    logger.info(f"[{artifact_key}] Loading: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    logger.info(f"[{artifact_key}] Raw shape: {df.shape}")
    logger.info(f"[{artifact_key}] Columns: {list(df.columns)}")

    # Identify text and label columns
    text_col = resolve_text_column(df)
    label_col = resolve_label_column(df)
    logger.info(f"[{artifact_key}] Text column: '{text_col}' | Label column: '{label_col}'")

    # ── Data cleansing ─────────────────────────────────────────────────────
    # Keep only rows where both text and label are non-null
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    dropped_null = before - len(df)
    if dropped_null:
        logger.info(f"[{artifact_key}] Dropped {dropped_null} rows with null text/label.")

    # ── Duplicate removal ──────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    dropped_dup = before - len(df)
    if dropped_dup:
        logger.info(f"[{artifact_key}] Dropped {dropped_dup} duplicate rows.")

    # ── Rename to standard column names ───────────────────────────────────
    df = df.rename(columns={text_col: "original_text", label_col: "class"})

    # ── Apply preprocessing pipeline ──────────────────────────────────────
    logger.info(f"[{artifact_key}] Preprocessing text...")
    df["cleaned_text"] = df["original_text"].apply(
        lambda t: preprocess_text(t, lemmatizer, stop_words)
    )

    # Drop rows where preprocessing results in empty string
    before = len(df)
    df = df[df["cleaned_text"].str.strip().ne("")]
    dropped_empty = before - len(df)
    if dropped_empty:
        logger.info(
            f"[{artifact_key}] Dropped {dropped_empty} rows with empty text after preprocessing."
        )

    # ── Binary label (for BiLSTM identification step) ─────────────────────
    df["binary_label"] = df["class"].apply(make_binary_label)

    df["category_label"] = df["class"].apply(categorize_label)

    # ── Column ordering ───────────────────────────────────────────────────
    cols = ["original_text", "cleaned_text", "class", "binary_label", "category_label"]
    extra_cols = [c for c in df.columns if c not in cols]
    df = df[cols + extra_cols]

    # ── Save output ───────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"preprocessed_{artifact_key.lower()}.csv"
    out_path = os.path.join(output_dir, out_filename)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"[{artifact_key}] Saved preprocessed data → {out_path}")

    # ── Label distribution summary ────────────────────────────────────────
    logger.info(f"[{artifact_key}] Label distribution (multi-class):")
    for label, count in df["class"].value_counts().items():
        logger.info(f"           {label:20s}: {count:6d}")
    logger.info(f"[{artifact_key}] Label distribution (binary):")
    for label, count in df["binary_label"].value_counts().items():
        logger.info(f"           {label:20s}: {count:6d}")
    logger.info(f"[{artifact_key}] Label distribution (category):")
    for label, count in df["category_label"].value_counts().items():
        logger.info(f"           {label:20s}: {count:6d}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Entry point for preprocessing all four SATD artifact datasets.

    Directory layout expected:
        satd_project/
            data/
                data-augmentation-code_comments.csv
                data-augmentation-commit-messages.csv
                data-augmentation-issues.csv
                data-augmentation-pull-requests.csv
            01_preprocessing.py

    Output:
        satd_project/
            data/
                preprocessed/
                    preprocessed_cc.csv
                    preprocessed_cm.csv
                    preprocessed_is.csv
                    preprocessed_ps.csv
    """
    # ── Paths ──────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(data_dir, "preprocessed")

    # ── NLTK setup ────────────────────────────────────────────────────────
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # ── Process each artifact ─────────────────────────────────────────────
    results = {}
    missing_files = []

    for artifact_key, filename in ARTIFACT_FILES.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(file_path):
            logger.warning(f"[{artifact_key}] File not found, skipping: {file_path}")
            missing_files.append(filename)
            continue
        df = process_artifact(artifact_key, file_path, output_dir, lemmatizer, stop_words)
        results[artifact_key] = df

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    for artifact_key, df in results.items():
        total = len(df)
        satd = (df["binary_label"] == "SATD").sum()
        not_satd = (df["binary_label"] == NOT_SATD_LABEL).sum()
        logger.info(
            f"  {artifact_key:4s} | Total: {total:6d} | SATD: {satd:6d} | Not-SATD: {not_satd:6d}"
        )

    if missing_files:
        logger.warning(f"Skipped (files not found): {missing_files}")

    if not results:
        logger.error(
            "No files were processed. Place the four CSV files in the 'data/' directory:\n"
            + "\n".join(f"  - {f}" for f in ARTIFACT_FILES.values())
        )
        return

    logger.info(f"Preprocessed files saved to: {output_dir}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
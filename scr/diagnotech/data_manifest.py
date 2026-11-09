"""Dataset manifests: multi-label encoding, validation, duplicate handling and leakage-safe splitting.

This module depends only on pandas, numpy and Pillow, so the data logic can be
unit-tested without PyTorch.

Manifests store *relative* image paths. Absolute paths are resolved against the
raw-data directory of the current session, so manifests stay valid after a Colab
runtime restart and the dataset is simply re-extracted to the same location.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

STATUS_OK = "ok"
STATUS_OUT_OF_SCOPE_ONLY = "out_of_scope_only"
STATUS_CONTRADICTORY = "contradictory"
STATUS_EMPTY = "empty"

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class LabelEncoding:
    """Multi-hot encoding of one raw annotation string plus its validation status."""

    vector: tuple[int, ...]
    status: str


@dataclass(frozen=True)
class ImageCheck:
    """Structural validation result for one image file."""

    path: str
    ok: bool
    width: int | None
    height: int | None
    mode: str | None
    error: str | None


def encode_finding_labels(
    raw_labels: Any,
    classes: Sequence[str],
    separator: str = "|",
    no_finding_token: str = "No Finding",
) -> LabelEncoding:
    """Encode an annotation string such as ``"Effusion|Mass"`` into a multi-hot vector.

    Rules:
      * Only tokens in ``classes`` set vector entries. Other findings (for example
        ``Pneumonia``) are outside the eight-class scope and are reported via status.
      * ``No Finding`` must not co-occur with any other label; such rows are flagged
        as ``contradictory`` rather than guessed at.
      * A row whose only findings are out of scope is flagged as ``out_of_scope_only``.
        It is never silently converted into ``No Finding``.
    """
    n = len(classes)
    if raw_labels is None or pd.isna(raw_labels):
        return LabelEncoding((0,) * n, STATUS_EMPTY)

    tokens = {tok.strip() for tok in str(raw_labels).split(separator) if tok.strip()}
    if not tokens:
        return LabelEncoding((0,) * n, STATUS_EMPTY)

    in_scope = tokens & set(classes)
    if not in_scope:
        return LabelEncoding((0,) * n, STATUS_OUT_OF_SCOPE_ONLY)
    if no_finding_token in tokens and len(tokens) > 1:
        return LabelEncoding((0,) * n, STATUS_CONTRADICTORY)

    vector = tuple(int(cls in in_scope) for cls in classes)
    return LabelEncoding(vector, STATUS_OK)


def build_manifest(
    annotations: pd.DataFrame,
    image_index: Mapping[str, str],
    annotation_cfg: Mapping[str, Any],
    classes: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row per usable image plus an audit dictionary describing every exclusion.

    Args:
        annotations: Raw annotation table (one row per image expected).
        image_index: Maps image file name -> path relative to the raw-data root.
        annotation_cfg: The ``annotation`` block of project_config.json.
        classes: Ordered target classes.
    """
    image_col = annotation_cfg["image_id_column"]
    label_col = annotation_cfg["label_column"]
    patient_col = annotation_cfg["patient_id_column"]
    separator = annotation_cfg.get("label_separator", "|")
    no_finding = annotation_cfg.get("no_finding_token", "No Finding")

    keep_statuses = {STATUS_OK}
    if annotation_cfg.get("keep_out_of_scope_only_as_negatives", False):
        keep_statuses.add(STATUS_OUT_OF_SCOPE_ONLY)

    required = [image_col, label_col, patient_col]
    absent = [col for col in required if col not in annotations.columns]
    if absent:
        raise ValueError(f"Annotation file is missing columns {absent}; found {list(annotations.columns)}")

    audit: dict[str, Any] = {"annotation_rows": int(len(annotations))}

    df = annotations[required].copy()
    df.columns = ["image_id", "finding_labels", "patient_id"]
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    duplicated = df["image_id"].duplicated(keep="first")
    audit["duplicate_annotation_rows_removed"] = int(duplicated.sum())
    df = df.loc[~duplicated]

    has_file = df["image_id"].map(lambda key: key in image_index)
    audit["annotated_images_without_file"] = int((~has_file).sum())
    df = df.loc[has_file].copy()
    audit["image_files_without_annotation"] = int(len(set(image_index) - set(df["image_id"])))

    encodings = [encode_finding_labels(raw, classes, separator, no_finding) for raw in df["finding_labels"]]
    statuses = pd.Series([enc.status for enc in encodings], index=df.index)
    audit["label_status_counts"] = {str(k): int(v) for k, v in statuses.value_counts().items()}

    keep = statuses.isin(keep_statuses).to_numpy()
    audit["excluded_rows"] = int((~keep).sum())

    kept_encodings = [enc for enc, flag in zip(encodings, keep) if flag]
    vectors = np.array([enc.vector for enc in kept_encodings], dtype=np.int64).reshape(-1, len(classes))

    df = df.loc[keep].copy()
    df["relative_path"] = df["image_id"].map(lambda key: image_index[key])
    df["label_status"] = [enc.status for enc in kept_encodings]
    for idx, cls in enumerate(classes):
        df[cls] = vectors[:, idx]
    df["num_positive_labels"] = vectors.sum(axis=1)

    ordered_columns = [
        "image_id", "relative_path", "patient_id", "finding_labels",
        "label_status", "num_positive_labels", *classes,
    ]
    df = df[ordered_columns].reset_index(drop=True)

    audit["manifest_rows"] = int(len(df))
    audit["unique_patients"] = int(df["patient_id"].nunique())
    audit["positives_per_class"] = {cls: int(df[cls].sum()) for cls in classes}
    return df, audit


def index_image_files(root: str | Path, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> tuple[dict[str, str], list[str]]:
    """Index every image under ``root`` by file name. Returns (name -> relative path, duplicate names).

    macOS ``__MACOSX`` resource forks in zip archives are ignored. If a file name
    appears twice, the first occurrence is kept and the name is reported.
    """
    root = Path(root)
    index: dict[str, str] = {}
    duplicates: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if "__MACOSX" in path.parts:
            continue
        if path.name in index:
            duplicates.append(path.name)
            continue
        index[path.name] = path.relative_to(root).as_posix()
    return index, duplicates


def resolve_image_paths(manifest: pd.DataFrame, raw_root: str | Path) -> pd.DataFrame:
    """Return a copy of the manifest with an absolute ``image_path`` column for this session."""
    root = Path(raw_root)
    out = manifest.copy()
    out["image_path"] = out["relative_path"].map(lambda rel: str(root / rel))
    return out


def inspect_image(path: str | Path) -> ImageCheck:
    """Verify an image file's integrity (CRC via PIL ``verify``) and record its geometry."""
    path = Path(path)
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
        return ImageCheck(str(path), True, width, height, mode, None)
    except Exception as exc:  # any decode failure marks the file as corrupted
        return ImageCheck(str(path), False, None, None, None, f"{type(exc).__name__}: {exc}")


def file_sha1(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Byte-level hash used for exact-duplicate detection (pixel-level duplicates are not detected)."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_leakage_groups(manifest: pd.DataFrame, patient_col: str = "patient_id", hash_col: str | None = None) -> pd.Series:
    """Assign each row to a group so rows that must not be separated stay together.

    Patients are linked (union-find) when they share an identical image file.
    Without this, one image duplicated across two patients could appear in both
    training and test data.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    patients = manifest[patient_col].astype(str)
    for patient in patients.unique():
        find(patient)

    if hash_col is not None and hash_col in manifest.columns:
        hashed = manifest[[hash_col, patient_col]].dropna()
        for _, group in hashed.groupby(hash_col)[patient_col]:
            members = group.astype(str).unique()
            for other in members[1:]:
                union(members[0], other)

    return patients.map(find)


def patient_level_split(
    manifest: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    group_col: str,
    classes: Sequence[str],
) -> pd.Series:
    """Assign each group to a split, targeting image-level fractions.

    Groups containing rare findings are placed first, so rare classes spread
    across splits instead of concentrating in one. Remaining groups are placed
    in random order (seeded). Each group goes to the split with the largest
    remaining deficit. The result is deterministic for a given seed.
    """
    names = list(fractions)
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be positive")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1, got {sum(fractions.values())}")

    rng = np.random.default_rng(seed)
    group_sizes = manifest.groupby(group_col).size()
    group_classes = manifest.groupby(group_col)[list(classes)].max()

    prevalence = manifest[list(classes)].mean(axis=0).clip(lower=1e-9)
    inverse_prevalence = (1.0 / prevalence).to_numpy(dtype=float)
    rarity = group_classes.to_numpy(dtype=float) @ inverse_prevalence
    tiebreak = rng.random(len(group_sizes))
    order = np.lexsort((tiebreak, -rarity))
    ordered_groups = group_sizes.index.to_numpy()[order]

    total = float(group_sizes.sum())
    targets = {name: fractions[name] * total for name in names}
    filled = {name: 0.0 for name in names}
    sizes = group_sizes.to_dict()
    assignment: dict[Any, str] = {}
    for group in ordered_groups:
        chosen = max(names, key=lambda name: targets[name] - filled[name])
        assignment[group] = chosen
        filled[chosen] += sizes[group]

    return manifest[group_col].map(assignment)


def verify_split(
    manifest: pd.DataFrame,
    split_col: str,
    group_col: str,
    classes: Sequence[str],
    min_positives: int,
) -> dict[str, Any]:
    """Assert leakage-freedom and summarise each split. Raises on leakage; warns on thin evaluation splits."""
    spans = manifest.groupby(group_col)[split_col].nunique()
    leaking_groups = int((spans > 1).sum())
    if leaking_groups:
        raise ValueError(f"Leakage: {leaking_groups} groups span more than one split.")

    patient_spans = manifest.groupby("patient_id")[split_col].nunique()
    if (patient_spans > 1).any():
        raise ValueError("Leakage: at least one patient appears in more than one split.")

    rows = manifest[split_col].value_counts()
    summary: dict[str, Any] = {
        "rows_per_split": {str(k): int(v) for k, v in rows.items()},
        "patients_per_split": {
            str(k): int(v) for k, v in manifest.groupby(split_col)["patient_id"].nunique().items()
        },
        "positives_per_split": {},
        "prevalence_per_split": {},
        "warnings": [],
    }
    for split in rows.index:
        subset = manifest[manifest[split_col] == split]
        counts = {cls: int(subset[cls].sum()) for cls in classes}
        summary["positives_per_split"][str(split)] = counts
        summary["prevalence_per_split"][str(split)] = {
            cls: round(counts[cls] / max(len(subset), 1), 6) for cls in classes
        }
        if split in ("val", "test"):
            for cls, value in counts.items():
                if value < min_positives:
                    summary["warnings"].append(
                        f"{split}: only {value} positives for '{cls}' (< {min_positives}); per-class metrics will be unstable"
                    )
    return summary


def class_statistics(train_df: pd.DataFrame, classes: Sequence[str], pos_weight_max: float | None = None) -> dict[str, Any]:
    """Compute class weights from the training split only.

    * ``pos_weight`` = negatives / positives, the standard inverse-frequency weight for
      independent (multi-label) BCE. Clipped at ``pos_weight_max`` to limit gradient spikes.
    * ``balanced_weight`` = N / (K * positives), the sklearn-style balanced weight, for reporting.
    """
    n = len(train_df)
    positives = train_df[list(classes)].sum().astype(int)
    if (positives == 0).any():
        missing = list(positives[positives == 0].index)
        raise ValueError(f"Classes with zero training positives: {missing}")
    negatives = n - positives
    pos_weight = (negatives / positives).astype(float)
    if pos_weight_max is not None:
        pos_weight = pos_weight.clip(upper=float(pos_weight_max))
    balanced = (n / (len(classes) * positives)).astype(float)
    return {
        "n_train": int(n),
        "positives": {c: int(positives[c]) for c in classes},
        "negatives": {c: int(negatives[c]) for c in classes},
        "prevalence": {c: round(float(positives[c]) / n, 6) for c in classes},
        "pos_weight": {c: round(float(pos_weight[c]), 6) for c in classes},
        "pos_weight_max": pos_weight_max,
        "balanced_weight": {c: round(float(balanced[c]), 6) for c in classes},
    }


def sampler_weights(df: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    """Per-image sampling weight: mean inverse training frequency of that image's positive labels.

    Images carrying rare findings are drawn more often. Images with no positive
    label (only possible if out-of-scope rows were kept) receive the mean weight.
    """
    matrix = df[list(classes)].to_numpy(dtype=float)
    positives = matrix.sum(axis=0)
    if (positives == 0).any():
        raise ValueError("Cannot build sampler weights: a class has zero positives")
    inverse = len(df) / positives
    row_sum = matrix.sum(axis=1)
    weighted = (matrix @ inverse) / np.maximum(row_sum, 1.0)
    fallback = float(inverse.mean())
    return np.where(row_sum > 0, weighted, fallback)


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id": str, "patient_id": str, "relative_path": str})

"""Configuration loading and storage-path management for DiagnoTech Model 2.

* The **code root** is the extracted project (contains ``src/``, ``config/``,
  ``notebooks/``). Configuration is always read from here.
* The **storage root** holds run artifacts (manifests, checkpoints, logs,
  reports). In Colab it is a folder on Google Drive; locally it is ``storage/``
  inside the code root, or the folder named by ``DIAGNOTECH_STORAGE_ROOT``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REQUIRED_CLASSES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pneumothorax",
)

REQUIRED_EXPERIMENT_KEYS: tuple[str, ...] = (
    "description", "augmentation", "dropout", "optimizer", "weight_decay",
    "fine_tune", "loss_weighting", "sampler", "label_smoothing",
    "scheduler", "early_stopping",
)

STORAGE_SUBDIRS: tuple[str, ...] = (
    "data", "config", "checkpoints", "evaluation",
    "experiments", "deployment", "logs", "reports",
)


def in_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def default_code_root() -> Path:
    """Project root, assuming this file lives at ``<root>/src/diagnotech/config.py``."""
    return Path(__file__).resolve().parents[2]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def load_json(path: str | Path) -> Any:
    """Read a JSON file, failing loudly if it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def config_fingerprint(*parts: Any) -> str:
    """Stable short hash of configuration objects, used to guard checkpoint resumption."""
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class StoragePaths:
    """Named sub-directories of the storage root (Drive folder in Colab)."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def deployment(self) -> Path:
        return self.root / "deployment"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "StoragePaths":
        """Create every sub-directory if missing (idempotent)."""
        for name in STORAGE_SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class ProjectContext:
    """Everything a notebook needs, loaded once at the top of the notebook."""

    code_root: Path
    storage: StoragePaths
    project: Mapping[str, Any]
    experiments: Mapping[str, Mapping[str, Any]]
    classes: tuple[str, ...]


def resolve_storage_root(project: Mapping[str, Any], override: str | Path | None = None) -> StoragePaths:
    """Choose and create the storage root: explicit override, env var, Drive (Colab), or local."""
    if override is not None:
        root = Path(override)
    elif os.environ.get("DIAGNOTECH_STORAGE_ROOT"):
        root = Path(os.environ["DIAGNOTECH_STORAGE_ROOT"])
    elif in_colab():
        mount = Path(project["source"]["colab_drive_mount"])
        if not mount.exists():
            raise RuntimeError(
                "Google Drive is not mounted. Run "
                "`from google.colab import drive; drive.mount('/content/drive')` first."
            )
        root = mount / project["source"]["storage_folder_name"]
    else:
        root = default_code_root() / "storage"
    return StoragePaths(root=root).ensure()


def load_context(code_root: str | Path | None = None, storage_root: str | Path | None = None) -> ProjectContext:
    """Load and validate configuration, resolve storage, and snapshot the config used by this run."""
    code_root = Path(code_root) if code_root is not None else default_code_root()
    config_dir = code_root / "config"

    project = load_json(config_dir / "project_config.json")
    experiments = load_json(config_dir / "experiments.json")
    class_map = load_json(config_dir / "class_mapping.json")

    classes = tuple(project["classes"])
    if classes != REQUIRED_CLASSES:
        raise ValueError(f"project_config.json classes must be exactly {REQUIRED_CLASSES}; found {classes}")
    if tuple(class_map["classes"]) != classes:
        raise ValueError("class_mapping.json disagrees with project_config.json; fix before running.")

    for exp_id, exp in experiments.items():
        missing = [k for k in REQUIRED_EXPERIMENT_KEYS if k not in exp]
        if missing:
            raise ValueError(f"Experiment {exp_id} is missing keys: {missing}")

    storage = resolve_storage_root(project, storage_root)
    for name in ("project_config.json", "experiments.json", "class_mapping.json"):
        shutil.copy2(config_dir / name, storage.config / name)

    return ProjectContext(
        code_root=code_root,
        storage=storage,
        project=project,
        experiments=experiments,
        classes=classes,
    )

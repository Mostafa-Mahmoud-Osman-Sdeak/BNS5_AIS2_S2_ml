"""Staged DenseNet121 training with validation-driven stopping and resumable checkpoints.

Only the ``train`` and ``val`` manifests are accepted. The ``test`` manifest belongs
to notebook 03 and is rejected here by design, which keeps final evaluation free of
leakage and of selection bias.

Resumption model
----------------
A checkpoint records the current stage, the epoch within that stage, the early-stopping
counter, the best validation loss, the full history, model/optimizer/scheduler/scaler
state and RNG state. Re-running the same experiment continues from the last completed
epoch. A checkpoint produced under a different configuration (see ``config_fingerprint``)
is refused rather than silently mixed with new settings.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from .config import ProjectContext, config_fingerprint, save_json
from .data_manifest import class_statistics
from .metrics import binarize, class_column_slug, exact_match_accuracy, hamming_accuracy, macro_auroc, macro_f1, per_class_report, tune_thresholds
from .model import build_densenet121, build_optimizer, count_parameters, set_train_mode, set_unfreeze_level
from .preprocessing import ChestXrayDataset, build_loader, build_transforms, build_weighted_sampler

logger = logging.getLogger("diagnotech.training")

TRAINING_SPLITS: tuple[str, ...] = ("train", "val")


class MultiLabelBCELoss(nn.Module):
    """Binary cross-entropy over independent labels, with optional ``pos_weight`` and label smoothing.

    Label smoothing maps targets y -> y * (1 - s) + s / 2, which softens the 0/1 targets
    for each independent label.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.register_buffer("pos_weight", pos_weight)  # moves with .to(device)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


@dataclass(frozen=True)
class Stage:
    """One contiguous block of epochs with fixed trainable layers and learning rates."""

    name: str
    unfreeze_level: int
    lr_head: float
    lr_backbone: Optional[float]
    max_epochs: int
    patience: Optional[int]
    use_scheduler: bool


def build_stage_plan(experiment: Mapping[str, Any], training_cfg: Mapping[str, Any]) -> list[Stage]:
    """Translate an experiment definition into an ordered list of training stages."""
    use_scheduler = bool(experiment["scheduler"])
    early_stopping = bool(experiment["early_stopping"])

    phase_a = training_cfg["phase_a"]
    stages = [
        Stage(
            name="phase_a_feature_extraction",
            unfreeze_level=0,
            lr_head=float(phase_a["lr_head"]),
            lr_backbone=None,
            max_epochs=int(phase_a["max_epochs"]),
            patience=int(phase_a["patience"]) if early_stopping else None,
            use_scheduler=use_scheduler,
        )
    ]

    if experiment["fine_tune"]:
        phase_b = training_cfg["phase_b"]
        for index, spec in enumerate(phase_b["stages"], start=1):
            level = int(spec["unfreeze_level"])
            stages.append(Stage(
                name=f"phase_b_stage{index}_unfreeze_level{level}",
                unfreeze_level=level,
                lr_head=float(phase_b["lr_head"]),
                lr_backbone=float(phase_b["lr_backbone"]),
                max_epochs=int(spec["max_epochs"]),
                patience=int(spec["patience"]) if early_stopping else None,
                use_scheduler=use_scheduler,
            ))
    return stages


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. cuDNN autotuning is enabled for speed, so runs are reproducible but not bit-identical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(enabled: bool):
    """Gradient scaler for mixed precision, compatible across PyTorch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _amp_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over the training loader. Returns (mean loss, targets, probabilities)."""
    set_train_mode(model)
    total_loss, total_items = 0.0, 0
    targets_all, probs_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=use_amp):
            logits = model(images)
        logits = logits.float()  # compute the loss in fp32 for numerical stability
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss. Halting before the checkpoint is overwritten.")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = images.size(0)
        total_loss += float(loss.detach()) * batch
        total_items += batch
        targets_all.append(targets.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())

    if total_items == 0:
        raise ValueError("The training loader produced no batches.")
    return total_loss / total_items, np.concatenate(targets_all), np.concatenate(probs_all)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = False,
    criterion: Optional[nn.Module] = None,
) -> tuple[Optional[float], np.ndarray, np.ndarray]:
    """Inference over a loader. Returns (loss or None, targets, probabilities). Order matches the manifest."""
    model.eval()
    total_loss, total_items = 0.0, 0
    targets_all, probs_all = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        if criterion is not None:
            loss = criterion(logits, targets.to(device, non_blocking=True))
            total_loss += float(loss) * images.size(0)
        total_items += images.size(0)
        targets_all.append(targets.numpy())
        probs_all.append(torch.sigmoid(logits).cpu().numpy())

    if total_items == 0:
        raise ValueError("The evaluation loader produced no batches.")
    loss_value = total_loss / total_items if criterion is not None else None
    return loss_value, np.concatenate(targets_all), np.concatenate(probs_all)


def summarize_epoch(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Epoch metrics at a fixed threshold, plus the threshold-free macro AUROC."""
    preds = binarize(y_prob, threshold)
    return {
        "label_accuracy": hamming_accuracy(y_true, preds),
        "macro_f1": macro_f1(y_true, preds),
        "macro_auroc": macro_auroc(y_true, y_prob),
    }


def _lr_snapshot(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {group.get("name", f"group{i}"): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def _gpu_peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_scheduler(optimizer: torch.optim.Optimizer, sched_cfg: Mapping[str, Any]) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg["mode"],
        factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]),
        min_lr=float(sched_cfg["min_lr"]),
    )


def _resume_payload(
    experiment_id: str,
    fingerprint: str,
    classes: list[str],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[ReduceLROnPlateau],
    scaler,
    progress: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "config_fingerprint": fingerprint,
        "class_mapping": list(classes),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "progress": dict(progress),
        "history": list(history),
        "rng_state": capture_rng_state(),
        "saved_at": _timestamp(),
    }


def _best_payload(
    experiment_id: str,
    fingerprint: str,
    classes: list[str],
    model: nn.Module,
    dropout: float,
    preprocessing: Mapping[str, Any],
    global_epoch: int,
    val_loss: float,
) -> dict[str, Any]:
    """Lightweight deployable payload: weights plus the metadata inference needs."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "config_fingerprint": fingerprint,
        "architecture": "densenet121",
        "dropout": float(dropout),
        "class_mapping": list(classes),
        "preprocessing": dict(preprocessing),
        "model_state": model.state_dict(),
        "best_epoch": int(global_epoch),
        "best_val_loss": float(val_loss),
        "saved_at": _timestamp(),
    }


def _upsert_results(path: Path, row: Mapping[str, Any]) -> None:
    """Insert or replace this experiment's row in experiment_results.csv."""
    new_row = pd.DataFrame([dict(row)])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["experiment_id"] != row["experiment_id"]]
        table = pd.concat([existing, new_row], ignore_index=True)
    else:
        table = new_row
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _validate_training_manifests(manifests: Mapping[str, pd.DataFrame]) -> None:
    unexpected = sorted(set(manifests) - set(TRAINING_SPLITS))
    if unexpected:
        raise ValueError(
            f"Training accepts only {TRAINING_SPLITS}; received {unexpected}. "
            "The test split is reserved for notebook 03 (final evaluation)."
        )
    missing = sorted(set(TRAINING_SPLITS) - set(manifests))
    if missing:
        raise ValueError(f"Missing required manifests: {missing}")


def _attach_logging(log_path: Path) -> list[logging.Handler]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    for handler in (file_handler, console_handler):
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return [file_handler, console_handler]


def _detach_logging(handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def run_experiment(
    experiment_id: str,
    ctx: ProjectContext,
    manifests: Mapping[str, pd.DataFrame],
    device: torch.device,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train (or resume) one experiment and return its validation summary.

    Artifacts:
        checkpoints/<ID>/best_model.pth, last_checkpoint.pth, training_history.csv, training_config.json
        experiments/experiment_results.csv (one row per experiment, upserted)
        experiments/<ID>_val_per_class.csv
        logs/<ID>_train.log
    """
    _validate_training_manifests(manifests)
    handlers = _attach_logging(ctx.storage.logs / f"{experiment_id}_train.log")
    try:
        return _run_experiment(experiment_id, ctx, manifests, device, batch_size, num_workers, resume)
    finally:
        _detach_logging(handlers)


def _run_experiment(
    experiment_id: str,
    ctx: ProjectContext,
    manifests: Mapping[str, pd.DataFrame],
    device: torch.device,
    batch_size: Optional[int],
    num_workers: Optional[int],
    resume: bool,
) -> dict[str, Any]:
    if experiment_id not in ctx.experiments:
        raise KeyError(f"Unknown experiment '{experiment_id}'. Known: {sorted(ctx.experiments)}")

    exp = dict(ctx.experiments[experiment_id])
    tcfg = ctx.project["training"]
    pcfg = ctx.project["preprocessing"]
    acfg = ctx.project["augmentation"]
    mcfg = ctx.project["model"]
    classes = list(ctx.classes)

    batch_size = int(batch_size if batch_size is not None else tcfg["batch_size"])
    if batch_size not in tcfg["allowed_batch_sizes"]:
        raise ValueError(f"batch_size must be one of {tcfg['allowed_batch_sizes']}, got {batch_size}")
    num_workers = int(num_workers if num_workers is not None else tcfg["num_workers"])
    seed = int(tcfg["seed"])
    seed_everything(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"

    train_df, val_df = manifests["train"], manifests["val"]
    stats = class_statistics(train_df, classes, pos_weight_max=tcfg["pos_weight_max"])

    ckpt_dir = ctx.storage.checkpoints / experiment_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last_checkpoint.pth"
    best_path = ckpt_dir / "best_model.pth"

    training_cfg_for_fingerprint = {k: v for k, v in tcfg.items() if k not in ("batch_size", "num_workers", "allowed_batch_sizes")}
    fingerprint = config_fingerprint(
        experiment_id,
        exp,
        mcfg,
        pcfg,
        acfg if exp["augmentation"] else None,
        training_cfg_for_fingerprint,
        classes,
        ctx.project["split"],
        sorted(train_df["image_id"]),
        sorted(val_df["image_id"]),
    )

    stages = build_stage_plan(exp, tcfg)

    pos_weight = None
    if exp["loss_weighting"] == "pos_weight":
        pos_weight = torch.tensor([stats["pos_weight"][c] for c in classes], dtype=torch.float32, device=device)
    elif exp["loss_weighting"] != "none":
        raise ValueError(f"Unknown loss_weighting '{exp['loss_weighting']}'")
    train_criterion = MultiLabelBCELoss(pos_weight, label_smoothing=float(exp["label_smoothing"])).to(device)
    # Validation loss is always unweighted and unsmoothed, so it is comparable across experiments.
    val_criterion = MultiLabelBCELoss(None, 0.0).to(device)

    model = build_densenet121(len(classes), dropout=float(exp["dropout"]), pretrained=bool(mcfg["pretrained"])).to(device)

    train_tf = build_transforms(pcfg, augment=bool(exp["augmentation"]), augmentation=acfg)
    eval_tf = build_transforms(pcfg, augment=False)
    if exp["sampler"] == "weighted_random":
        sampler = build_weighted_sampler(train_df, classes, seed)
    elif exp["sampler"] == "none":
        sampler = None
    else:
        raise ValueError(f"Unknown sampler '{exp['sampler']}'")

    pin = device.type == "cuda"
    train_loader = build_loader(
        ChestXrayDataset(train_df, classes, train_tf), batch_size,
        shuffle=True, sampler=sampler, num_workers=num_workers, pin_memory=pin, seed=seed,
    )
    val_loader = build_loader(
        ChestXrayDataset(val_df, classes, eval_tf), batch_size,
        shuffle=False, sampler=None, num_workers=num_workers, pin_memory=pin, seed=seed,
    )

    progress: dict[str, Any] = {
        "stage_index": 0,
        "epoch_in_stage": 0,
        "stale_epochs": 0,
        "global_epoch": 0,
        "best_val_loss": math.inf,
        "best_epoch": None,
        "elapsed_seconds": 0.0,
        "status": "running",
    }
    history: list[dict[str, Any]] = []
    resume_ckpt: Optional[dict[str, Any]] = None

    if resume and last_path.exists():
        resume_ckpt = load_checkpoint(last_path, map_location="cpu")
        if resume_ckpt["config_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"The checkpoint for {experiment_id} was produced under a different configuration. "
                f"Refusing to resume. Restore the original configuration, or move {ckpt_dir} aside to start over."
            )
        model.load_state_dict(resume_ckpt["model_state"])
        progress = dict(resume_ckpt["progress"])
        history = list(resume_ckpt["history"])
        restore_rng_state(resume_ckpt["rng_state"])
        logger.info(
            "Resuming %s: stage %d, epoch %d within stage, global epoch %d, status %s",
            experiment_id, progress["stage_index"], progress["epoch_in_stage"],
            progress["global_epoch"], progress["status"],
        )
    else:
        logger.info(
            "Starting %s from scratch | batch %d | workers %d | AMP %s | device %s",
            experiment_id, batch_size, num_workers, use_amp, device,
        )

    run_meta = {
        "experiment_id": experiment_id,
        "experiment": exp,
        "fingerprint": fingerprint,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "amp": use_amp,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "stages": [asdict(stage) for stage in stages],
        "parameters": count_parameters(model),
        "model": dict(mcfg),
        "preprocessing": dict(pcfg),
        "augmentation": dict(acfg) if exp["augmentation"] else None,
        "training": training_cfg_for_fingerprint,
        "classes": classes,
        "class_statistics": stats,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df))},
    }
    save_json(run_meta, ckpt_dir / "training_config.json")

    scaler = make_grad_scaler(use_amp)
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[ReduceLROnPlateau] = None

    for stage_index in range(progress["stage_index"], len(stages)):
        stage = stages[stage_index]
        mid_stage_resume = (
            resume_ckpt is not None
            and stage_index == progress["stage_index"]
            and progress["epoch_in_stage"] > 0
        )

        set_unfreeze_level(model, stage.unfreeze_level)
        optimizer = build_optimizer(
            model, stage.lr_head, stage.lr_backbone, float(exp["weight_decay"]), exp["optimizer"],
        )
        scheduler = _make_scheduler(optimizer, tcfg["scheduler"]) if stage.use_scheduler else None

        start_epoch, stale = 0, 0
        if mid_stage_resume:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if scheduler is not None and resume_ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            if resume_ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(resume_ckpt["scaler_state"])
            start_epoch = int(progress["epoch_in_stage"])
            stale = int(progress["stale_epochs"])
            logger.info("Restored optimizer/scheduler state for %s at epoch %d", stage.name, start_epoch)
        resume_ckpt = None  # only the first stage encountered can be a mid-stage resume

        logger.info(
            "== %s | max %d epochs | lr_head %.1e | lr_backbone %s | patience %s | trainable params %s",
            stage.name, stage.max_epochs, stage.lr_head,
            f"{stage.lr_backbone:.1e}" if stage.lr_backbone else "n/a",
            stage.patience, f"{count_parameters(model)['trainable']:,}",
        )

        for epoch_in_stage in range(start_epoch, stage.max_epochs):
            if stage.patience is not None and stale >= stage.patience:
                logger.info("Early stopping in %s (no validation improvement for %d epochs)", stage.name, stale)
                break

            epoch_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr_used = _lr_snapshot(optimizer)

            train_loss, tr_true, tr_prob = train_one_epoch(
                model, train_loader, train_criterion, optimizer, scaler, device, use_amp,
            )
            val_loss, va_true, va_prob = collect_predictions(
                model, val_loader, device, use_amp=use_amp, criterion=val_criterion,
            )
            epoch_seconds = time.time() - epoch_start

            if scheduler is not None:
                scheduler.step(val_loss)

            progress["global_epoch"] += 1
            progress["elapsed_seconds"] += epoch_seconds
            improved = val_loss < progress["best_val_loss"]
            if improved:
                progress["best_val_loss"] = float(val_loss)
                progress["best_epoch"] = int(progress["global_epoch"])
                stale = 0
                save_checkpoint_atomic(
                    _best_payload(experiment_id, fingerprint, classes, model, exp["dropout"],
                                  pcfg, progress["global_epoch"], val_loss),
                    best_path,
                )
            else:
                stale += 1

            tr_metrics = summarize_epoch(tr_true, tr_prob)
            va_metrics = summarize_epoch(va_true, va_prob)
            record = {
                "global_epoch": int(progress["global_epoch"]),
                "stage": stage.name,
                "epoch_in_stage": epoch_in_stage + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_label_accuracy": tr_metrics["label_accuracy"],
                "val_label_accuracy": va_metrics["label_accuracy"],
                "train_macro_f1": tr_metrics["macro_f1"],
                "val_macro_f1": va_metrics["macro_f1"],
                "train_macro_auroc": tr_metrics["macro_auroc"],
                "val_macro_auroc": va_metrics["macro_auroc"],
                "lr_head": lr_used.get("head"),
                "lr_backbone": lr_used.get("backbone"),
                "epoch_seconds": round(epoch_seconds, 2),
                "gpu_peak_memory_gb": round(_gpu_peak_gb(device), 3),
                "improved": bool(improved),
            }
            history.append(record)

            progress.update(stage_index=stage_index, epoch_in_stage=epoch_in_stage + 1, stale_epochs=stale)
            save_checkpoint_atomic(
                _resume_payload(experiment_id, fingerprint, classes, model, optimizer, scheduler, scaler, progress, history),
                last_path,
            )
            pd.DataFrame(history).to_csv(ckpt_dir / "training_history.csv", index=False)

            logger.info(
                "[%s] ep %3d | train %.4f | val %.4f | val_auroc %.4f | val_f1 %.4f | lr_head %.1e | %.0fs%s",
                experiment_id, progress["global_epoch"], train_loss, val_loss,
                va_metrics["macro_auroc"], va_metrics["macro_f1"], lr_used.get("head", float("nan")),
                epoch_seconds, " *best*" if improved else "",
            )

        progress.update(stage_index=stage_index + 1, epoch_in_stage=0, stale_epochs=0)
        save_checkpoint_atomic(
            _resume_payload(experiment_id, fingerprint, classes, model, optimizer, scheduler, scaler, progress, history),
            last_path,
        )

    progress["status"] = "finished"
    save_checkpoint_atomic(
        _resume_payload(experiment_id, fingerprint, classes, model, optimizer, scheduler, scaler, progress, history),
        last_path,
    )

    # Final validation report uses the best weights, never the last-epoch weights.
    best = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    val_loss_final, va_true, va_prob = collect_predictions(model, val_loader, device, use_amp=use_amp, criterion=val_criterion)
    thresholds = tune_thresholds(va_true, va_prob)
    report = per_class_report(va_true, va_prob, classes, thresholds=thresholds)
    report.insert(0, "experiment_id", experiment_id)
    report.to_csv(ctx.storage.experiments / f"{experiment_id}_val_per_class.csv", index=False)

    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "description": exp["description"],
        "batch_size": batch_size,
        "best_epoch": progress["best_epoch"],
        "epochs_trained": progress["global_epoch"],
        "val_loss": float(val_loss_final),
        "val_label_accuracy": hamming_accuracy(va_true, binarize(va_prob, 0.5)),
        "val_exact_match": exact_match_accuracy(va_true, binarize(va_prob, 0.5)),
        "val_macro_auroc": macro_auroc(va_true, va_prob),
        "val_macro_f1_default": macro_f1(va_true, binarize(va_prob, 0.5)),
        "val_macro_f1_tuned": macro_f1(va_true, binarize(va_prob, thresholds)),
        "min_class_auroc": float(np.nanmin(report["auroc"].to_numpy(dtype=float))),
        "training_seconds": round(float(progress["elapsed_seconds"]), 1),
    }
    for _, row in report.iterrows():
        summary[f"auroc_{class_column_slug(row['class'])}"] = float(row["auroc"])

    _upsert_results(ctx.storage.experiments / "experiment_results.csv", summary)
    logger.info(
        "[%s] finished | best epoch %s | val_loss %.4f | val_macro_auroc %.4f | val_macro_f1 (tuned) %.4f",
        experiment_id, summary["best_epoch"], summary["val_loss"],
        summary["val_macro_auroc"], summary["val_macro_f1_tuned"],
    )
    return summary

"""End-to-end torch-free integration test of the Notebook 01 data path, on synthetic data.

It creates an NIH-style image tree and annotation CSV, runs every data step, saves the
manifests and reloads them. It does not touch the real dataset.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnotech.config import REQUIRED_CLASSES, load_context  # noqa: E402
from diagnotech.data_manifest import (  # noqa: E402
    build_leakage_groups,
    build_manifest,
    class_statistics,
    file_sha1,
    index_image_files,
    inspect_image,
    load_manifest,
    patient_level_split,
    resolve_image_paths,
    sampler_weights,
    save_manifest,
    verify_split,
)


def _make_synthetic_archive(raw: Path, n_patients: int = 120, seed: int = 0):
    """Create images and an annotation CSV with the NIH column layout."""
    rng = np.random.default_rng(seed)
    label_pool = [c for c in REQUIRED_CLASSES if c != "No Finding"] + ["No Finding", "Pneumonia"]
    rows = []
    (raw / "images_001" / "images").mkdir(parents=True)
    for pid in range(n_patients):
        for visit in range(int(rng.integers(1, 4))):
            name = f"P{pid:04d}_V{visit}.png"
            Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8)).save(raw / "images_001" / "images" / name)
            k = int(rng.integers(1, 3))
            picks = rng.choice(label_pool, size=k, replace=False)
            if "No Finding" in picks and len(picks) > 1:
                picks = np.array(["No Finding"])
            rows.append({"Image Index": name, "Finding Labels": "|".join(picks), "Patient ID": pid})
    # Guarantee roughly a dozen single-label positives per class, so every class
    # lands in the training split after the patient-level split.
    for i in range(min(len(rows), 12 * len(REQUIRED_CLASSES))):
        rows[i]["Finding Labels"] = REQUIRED_CLASSES[i % len(REQUIRED_CLASSES)]
    # Add a corrupt image and an annotation with no file
    (raw / "images_001" / "images" / "CORRUPT.png").write_bytes(b"garbage")
    rows.append({"Image Index": "CORRUPT.png", "Finding Labels": "Mass", "Patient ID": 9999})
    rows.append({"Image Index": "MISSING.png", "Finding Labels": "Mass", "Patient ID": 9998})
    pd.DataFrame(rows).to_csv(raw / "Data_Entry_2017.csv", index=False)


class PipelineIntegrationTests(unittest.TestCase):
    def test_full_data_path_on_synthetic_archive(self):
        cfg = json.loads((ROOT / "config" / "project_config.json").read_text())
        ann_cfg = cfg["annotation"]
        classes = list(REQUIRED_CLASSES)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            _make_synthetic_archive(raw)

            annotations = pd.read_csv(raw / "Data_Entry_2017.csv")
            index, dups = index_image_files(raw)
            self.assertEqual(dups, [])

            manifest, audit = build_manifest(annotations, index, ann_cfg, classes)
            self.assertIn("Pneumonia", annotations["Finding Labels"].str.cat(sep="|"))
            self.assertGreaterEqual(audit["annotated_images_without_file"], 1)

            absolute = resolve_image_paths(manifest, raw)
            checks = [inspect_image(p) for p in absolute["image_path"]]
            ok = np.array([c.ok for c in checks])
            self.assertEqual(int((~ok).sum()), 1)  # CORRUPT.png only
            manifest = manifest.loc[ok].reset_index(drop=True)

            manifest["sha1"] = [file_sha1(p) for p in resolve_image_paths(manifest, raw)["image_path"]]
            manifest["group_id"] = build_leakage_groups(manifest, "patient_id", hash_col="sha1")
            fractions = {"train": cfg["split"]["train"], "val": cfg["split"]["val"], "test": cfg["split"]["test"]}
            manifest["split"] = patient_level_split(manifest, fractions, cfg["split"]["seed"], "group_id", classes)
            summary = verify_split(manifest, "split", "group_id", classes, min_positives=1)
            self.assertEqual(set(summary["rows_per_split"]), {"train", "val", "test"})

            storage = tmp / "storage" / "data"
            for split in ("train", "val", "test"):
                save_manifest(manifest.loc[manifest["split"] == split].reset_index(drop=True), storage / f"manifest_{split}.csv")
            reloaded = load_manifest(storage / "manifest_train.csv")
            self.assertEqual(len(reloaded), int((manifest["split"] == "train").sum()))
            self.assertIsInstance(reloaded["image_id"].iloc[0], str)
            # Relative paths resolve against whatever raw root the session uses
            resolved = resolve_image_paths(reloaded, raw)
            self.assertTrue(all(Path(p).exists() for p in resolved["image_path"]))

            train = manifest.loc[manifest["split"] == "train"]
            stats = class_statistics(train, classes, pos_weight_max=cfg["training"]["pos_weight_max"])
            self.assertEqual(stats["n_train"], len(train))
            self.assertTrue(all(0 < v <= cfg["training"]["pos_weight_max"] for v in stats["pos_weight"].values()))
            weights = sampler_weights(train, classes)
            self.assertEqual(len(weights), len(train))
            self.assertTrue(np.all(weights > 0))

    def test_context_loads_and_snapshots_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = load_context(ROOT, storage_root=Path(tmp) / "drive_sim")
            self.assertEqual(tuple(ctx.classes), tuple(REQUIRED_CLASSES))
            self.assertEqual(len(ctx.experiments), 9)
            for name in ("project_config.json", "experiments.json", "class_mapping.json"):
                self.assertTrue((ctx.storage.config / name).exists())
            for sub in ("data", "checkpoints", "experiments", "deployment", "logs", "reports", "evaluation"):
                self.assertTrue((ctx.storage.root / sub).is_dir(), sub)

    def test_experiment_definitions_follow_sequence(self):
        exps = json.loads((ROOT / "config" / "experiments.json").read_text())
        self.assertEqual(list(exps), [f"E{i}" for i in range(9)])
        self.assertFalse(exps["E0"]["augmentation"])
        self.assertTrue(exps["E1"]["augmentation"])
        self.assertEqual(exps["E2"]["dropout"], 0.3)
        self.assertEqual(exps["E3"]["optimizer"], "adamw")
        self.assertTrue(exps["E4"]["fine_tune"])
        self.assertEqual(exps["E5"]["loss_weighting"], "pos_weight")
        self.assertTrue(exps["E6"]["scheduler"] and exps["E6"]["early_stopping"])
        self.assertEqual(exps["E7"]["sampler"], "weighted_random")
        self.assertEqual(exps["E8"]["label_smoothing"], 0.05)

    def test_class_mapping_matches_project(self):
        mapping = json.loads((ROOT / "config" / "class_mapping.json").read_text())
        self.assertEqual(mapping["classes"], list(REQUIRED_CLASSES))
        self.assertEqual(mapping["class_to_index"]["No Finding"], 5)
        self.assertEqual(len(mapping["classes"]), 8)


if __name__ == "__main__":
    unittest.main()

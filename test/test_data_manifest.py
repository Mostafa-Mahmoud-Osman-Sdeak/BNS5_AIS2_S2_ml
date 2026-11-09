"""Torch-free tests for the data layer. Run: python -m unittest discover -s tests -v"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnotech.data_manifest import (  # noqa: E402
    STATUS_CONTRADICTORY,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_OUT_OF_SCOPE_ONLY,
    build_leakage_groups,
    build_manifest,
    class_statistics,
    encode_finding_labels,
    index_image_files,
    inspect_image,
    patient_level_split,
    sampler_weights,
    verify_split,
)

CLASSES = ("Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
           "Mass", "No Finding", "Nodule", "Pneumothorax")
ANNOTATION_CFG = {
    "image_id_column": "Image Index",
    "label_column": "Finding Labels",
    "patient_id_column": "Patient ID",
    "label_separator": "|",
    "no_finding_token": "No Finding",
    "keep_out_of_scope_only_as_negatives": False,
}


class LabelEncodingTests(unittest.TestCase):
    def test_multi_label_vector(self):
        enc = encode_finding_labels("Effusion|Mass", CLASSES)
        self.assertEqual(enc.status, STATUS_OK)
        self.assertEqual(enc.vector, (0, 0, 1, 0, 1, 0, 0, 0))

    def test_no_finding_only(self):
        enc = encode_finding_labels("No Finding", CLASSES)
        self.assertEqual(enc.status, STATUS_OK)
        self.assertEqual(enc.vector[CLASSES.index("No Finding")], 1)
        self.assertEqual(sum(enc.vector), 1)

    def test_out_of_scope_only_is_never_converted_to_no_finding(self):
        enc = encode_finding_labels("Pneumonia", CLASSES)
        self.assertEqual(enc.status, STATUS_OUT_OF_SCOPE_ONLY)
        self.assertEqual(sum(enc.vector), 0)

    def test_mixed_in_and_out_of_scope_keeps_in_scope_positive(self):
        enc = encode_finding_labels("Effusion|Pneumonia", CLASSES)
        self.assertEqual(enc.status, STATUS_OK)
        self.assertEqual(enc.vector[CLASSES.index("Effusion")], 1)
        self.assertEqual(sum(enc.vector), 1)

    def test_no_finding_with_disease_is_contradictory(self):
        self.assertEqual(encode_finding_labels("No Finding|Mass", CLASSES).status, STATUS_CONTRADICTORY)

    def test_empty_and_missing(self):
        self.assertEqual(encode_finding_labels("", CLASSES).status, STATUS_EMPTY)
        self.assertEqual(encode_finding_labels(None, CLASSES).status, STATUS_EMPTY)
        self.assertEqual(encode_finding_labels(float("nan"), CLASSES).status, STATUS_EMPTY)

    def test_duplicate_tokens_collapse(self):
        self.assertEqual(encode_finding_labels("Mass|Mass", CLASSES).vector[CLASSES.index("Mass")], 1)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.annotations = pd.DataFrame({
            "Image Index": ["a.png", "b.png", "c.png", "d.png", "e.png", "a.png", "f.png"],
            "Finding Labels": ["Effusion|Mass", "No Finding", "Pneumonia", "No Finding|Mass",
                               "Nodule", "Effusion|Mass", "Atelectasis"],
            "Patient ID": [1, 2, 3, 4, 5, 1, 6],
        })
        # g.png is on disk but has no annotation row. f.png is annotated but has no file.
        self.index = {n: f"images/{n}" for n in ["a.png", "b.png", "c.png", "d.png", "e.png", "g.png"]}

    def test_manifest_audit_accounts_for_every_row(self):
        manifest, audit = build_manifest(self.annotations, self.index, ANNOTATION_CFG, CLASSES)
        self.assertEqual(audit["duplicate_annotation_rows_removed"], 1)
        self.assertEqual(audit["annotated_images_without_file"], 1)
        self.assertEqual(audit["image_files_without_annotation"], 1)
        self.assertEqual(audit["label_status_counts"][STATUS_CONTRADICTORY], 1)
        self.assertEqual(audit["label_status_counts"][STATUS_OUT_OF_SCOPE_ONLY], 1)
        self.assertEqual(list(manifest["image_id"]), ["a.png", "b.png", "e.png"])
        self.assertTrue((manifest["num_positive_labels"] >= 1).all())

    def test_manifest_columns_and_patient_type(self):
        manifest, _ = build_manifest(self.annotations, self.index, ANNOTATION_CFG, CLASSES)
        for cls in CLASSES:
            self.assertIn(cls, manifest.columns)
        self.assertEqual(manifest.loc[manifest.image_id == "a.png", "Effusion"].item(), 1)
        self.assertEqual(str(manifest.loc[manifest.image_id == "a.png", "patient_id"].item()), "1")

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            build_manifest(self.annotations.drop(columns=["Patient ID"]), self.index, ANNOTATION_CFG, CLASSES)


class LeakageAndSplitTests(unittest.TestCase):
    @staticmethod
    def synthetic_manifest(n_patients=400, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for pid in range(n_patients):
            for _ in range(int(rng.integers(1, 6))):
                vec = (rng.random(len(CLASSES)) < 0.08).astype(int)
                if vec.sum() == 0:
                    vec[CLASSES.index("No Finding")] = 1
                rows.append({"image_id": f"img_{len(rows)}.png", "patient_id": f"p{pid}",
                             "sha1": f"h{len(rows)}", **dict(zip(CLASSES, vec))})
        return pd.DataFrame(rows)

    def test_shared_image_hash_links_patients(self):
        df = pd.DataFrame({"patient_id": ["p1", "p2", "p3"], "sha1": ["same", "same", "other"]})
        groups = build_leakage_groups(df, "patient_id", hash_col="sha1")
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])

    def test_split_has_no_leakage_and_hits_target_fractions(self):
        df = self.synthetic_manifest()
        df["group_id"] = build_leakage_groups(df, "patient_id", hash_col="sha1")
        fractions = {"train": 0.70, "val": 0.15, "test": 0.15}
        df["split"] = patient_level_split(df, fractions, seed=42, group_col="group_id", classes=CLASSES)
        report = verify_split(df, "split", "group_id", CLASSES, min_positives=1)  # raises on leakage

        total = len(df)
        for name, target in fractions.items():
            achieved = (df["split"] == name).sum() / total
            self.assertAlmostEqual(achieved, target, delta=0.03, msg=f"{name} fraction {achieved:.3f}")
        self.assertEqual(report["warnings"], [])

    def test_split_is_deterministic_for_seed(self):
        df = self.synthetic_manifest()
        fractions = {"train": 0.70, "val": 0.15, "test": 0.15}
        a = patient_level_split(df, fractions, 7, "patient_id", CLASSES)
        b = patient_level_split(df, fractions, 7, "patient_id", CLASSES)
        pd.testing.assert_series_equal(a, b)

    def test_verify_split_detects_leakage(self):
        df = pd.DataFrame({"patient_id": ["p1", "p1"], "group_id": ["g", "g"], "split": ["train", "test"],
                           **{c: [1, 0] for c in CLASSES}})
        with self.assertRaises(ValueError):
            verify_split(df, "split", "group_id", CLASSES, min_positives=1)

    def test_rare_class_present_in_every_split(self):
        df = self.synthetic_manifest(n_patients=600)
        df["group_id"] = df["patient_id"]
        df["split"] = patient_level_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, 42, "group_id", CLASSES)
        counts = df.groupby("split")["Pneumothorax"].sum()
        self.assertTrue((counts > 0).all(), counts.to_dict())


class ClassStatisticsTests(unittest.TestCase):
    def test_pos_weight_is_negatives_over_positives(self):
        # 4 rows. Mass: 1 positive -> pos_weight 3/1. Atelectasis: 2 positives -> 2/2.
        # Every other class has 1 positive, so no class has zero positives.
        df = pd.DataFrame(0, index=range(4), columns=list(CLASSES))
        df.loc[0, "Mass"] = 1
        df.loc[[0, 1], "Atelectasis"] = 1
        for cls in CLASSES:
            if df[cls].sum() == 0:
                df.loc[3, cls] = 1
        stats = class_statistics(df, CLASSES, pos_weight_max=None)
        self.assertAlmostEqual(stats["pos_weight"]["Atelectasis"], 1.0)
        self.assertAlmostEqual(stats["pos_weight"]["Mass"], 3.0)

    def test_pos_weight_is_clipped(self):
        df = pd.DataFrame({c: [0] * 99 + [1] for c in CLASSES})
        stats = class_statistics(df, CLASSES, pos_weight_max=50.0)
        self.assertEqual(stats["pos_weight"]["Mass"], 50.0)

    def test_zero_positive_class_raises(self):
        df = pd.DataFrame({c: [0, 0] for c in CLASSES})
        with self.assertRaises(ValueError):
            class_statistics(df, CLASSES)

    def test_sampler_favours_rare_label_images(self):
        # 10 rows. No Finding: rows 0-2 (3 positives, common).
        # Pneumothorax: row 3 only (1 positive, rare).
        # Six other classes: one row each, so every class has at least one positive.
        df = pd.DataFrame(0, index=range(10), columns=list(CLASSES))
        df.loc[0:2, "No Finding"] = 1
        df.loc[3, "Pneumothorax"] = 1
        other = [c for c in CLASSES if c not in ("No Finding", "Pneumothorax")]
        for offset, cls in enumerate(other):
            df.loc[4 + offset, cls] = 1
        w = sampler_weights(df, CLASSES)
        self.assertGreater(w[3], w[0])  # rare-label image drawn more often


class ImageIntegrityTests(unittest.TestCase):
    def test_valid_and_corrupt_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = tmp / "good.png"
            Image.new("L", (32, 16), color=128).save(good)
            bad = tmp / "bad.png"
            bad.write_bytes(b"not a real png file at all")

            ok = inspect_image(good)
            self.assertTrue(ok.ok)
            self.assertEqual((ok.width, ok.height), (32, 16))
            self.assertFalse(inspect_image(bad).ok)

    def test_index_ignores_macos_resource_forks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "images").mkdir()
            (tmp / "__MACOSX").mkdir()
            Image.new("L", (4, 4)).save(tmp / "images" / "x.png")
            Image.new("L", (4, 4)).save(tmp / "__MACOSX" / "x.png")
            index, dups = index_image_files(tmp)
            self.assertEqual(index, {"x.png": "images/x.png"})
            self.assertEqual(dups, [])


if __name__ == "__main__":
    unittest.main()

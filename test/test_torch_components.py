"""Torch-dependent tests. Skipped automatically when PyTorch/torchvision are not installed.

Run in Colab or any environment with torch:  python -m unittest discover -s tests -v
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None

if TORCH_AVAILABLE:
    import torch
    from diagnotech.checkpointing import load_checkpoint, save_checkpoint_atomic
    from diagnotech.model import (
        build_densenet121,
        build_optimizer,
        count_parameters,
        set_train_mode,
        set_unfreeze_level,
    )
    from diagnotech.preprocessing import ChestXrayDataset, build_loader, build_transforms
    from diagnotech.training import MultiLabelBCELoss, build_stage_plan, collect_predictions, make_grad_scaler, train_one_epoch

CLASSES = ("Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
           "Mass", "No Finding", "Nodule", "Pneumothorax")
PREPROCESSING = {"image_size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
AUGMENTATION = {"rotation_degrees": 5, "translate_fraction": 0.04, "rrc_scale": [0.9, 1.0],
                "rrc_ratio": [0.95, 1.05], "brightness": 0.1, "contrast": 0.1}


@unittest.skipUnless(TORCH_AVAILABLE, "torch/torchvision not installed in this environment")
class ModelTests(unittest.TestCase):
    def test_output_is_eight_independent_logits(self):
        model = build_densenet121(8, dropout=0.3, pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_dropout_is_in_head(self):
        model = build_densenet121(8, dropout=0.3, pretrained=False)
        self.assertIsInstance(model.classifier[0], torch.nn.Dropout)
        self.assertAlmostEqual(model.classifier[0].p, 0.3)

    def test_unfreeze_levels_are_cumulative(self):
        model = build_densenet121(8, dropout=0.0, pretrained=False)
        set_unfreeze_level(model, 0)
        head_only = count_parameters(model)["trainable"]
        set_unfreeze_level(model, 1)
        level1 = count_parameters(model)["trainable"]
        set_unfreeze_level(model, 2)
        level2 = count_parameters(model)["trainable"]
        self.assertLess(head_only, level1)
        self.assertLess(level1, level2)
        self.assertTrue(all(p.requires_grad for p in model.features.denseblock4.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.features.conv0.parameters()))

    def test_optimizer_has_separate_head_and_backbone_groups(self):
        model = build_densenet121(8, dropout=0.0, pretrained=False)
        set_unfreeze_level(model, 0)
        opt = build_optimizer(model, 1e-3, None, 0.0, "adam")
        self.assertEqual(len(opt.param_groups), 1)
        set_unfreeze_level(model, 1)
        opt = build_optimizer(model, 5e-4, 1e-4, 1e-4, "adamw")
        self.assertEqual([g["name"] for g in opt.param_groups], ["head", "backbone"])
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1e-4)

    def test_frozen_batchnorm_stays_in_eval_mode(self):
        model = build_densenet121(8, dropout=0.0, pretrained=False)
        set_unfreeze_level(model, 0)
        set_train_mode(model)
        self.assertFalse(model.features.norm0.training)
        self.assertTrue(model.classifier.training)

    def test_stage_plan_matches_experiment_flags(self):
        training = {"phase_a": {"max_epochs": 10, "lr_head": 1e-3, "patience": 3},
                    "phase_b": {"lr_head": 5e-4, "lr_backbone": 1e-4,
                                "stages": [{"unfreeze_level": 1, "max_epochs": 15, "patience": 4},
                                           {"unfreeze_level": 2, "max_epochs": 15, "patience": 4}]}}
        e0 = {"scheduler": False, "early_stopping": False, "fine_tune": False}
        e6 = {"scheduler": True, "early_stopping": True, "fine_tune": True}
        self.assertEqual(len(build_stage_plan(e0, training)), 1)
        plan = build_stage_plan(e6, training)
        self.assertEqual(len(plan), 3)
        self.assertIsNone(build_stage_plan(e0, training)[0].patience)
        self.assertEqual(sum(s.max_epochs for s in plan), 40)
        self.assertLessEqual(sum(s.max_epochs for s in plan[1:]), 30)


@unittest.skipUnless(TORCH_AVAILABLE, "torch/torchvision not installed in this environment")
class LossAndTransformTests(unittest.TestCase):
    def test_label_smoothing_changes_loss(self):
        logits = torch.tensor([[2.0, -2.0]])
        targets = torch.tensor([[1.0, 0.0]])
        plain = MultiLabelBCELoss(None, 0.0)(logits, targets)
        smoothed = MultiLabelBCELoss(None, 0.05)(logits, targets)
        self.assertNotAlmostEqual(float(plain), float(smoothed), places=6)

    def test_pos_weight_scales_positive_term(self):
        logits = torch.tensor([[-1.0]])
        targets = torch.tensor([[1.0]])
        base = float(MultiLabelBCELoss(None)(logits, targets))
        weighted = float(MultiLabelBCELoss(torch.tensor([4.0]))(logits, targets))
        self.assertAlmostEqual(weighted, 4.0 * base, places=5)

    def test_transform_shape_and_no_flip(self):
        tf = build_transforms(PREPROCESSING, augment=True, augmentation=AUGMENTATION)
        image = Image.fromarray((np.arange(128 * 128) % 255).reshape(128, 128).astype(np.uint8)).convert("RGB")
        out = tf(image)
        self.assertEqual(tuple(out.shape), (3, 64, 64))
        # Eval transform is deterministic
        eval_tf = build_transforms(PREPROCESSING, augment=False)
        self.assertTrue(torch.equal(eval_tf(image), eval_tf(image)))

    def test_augmentation_requires_config(self):
        with self.assertRaises(ValueError):
            build_transforms(PREPROCESSING, augment=True, augmentation=None)


@unittest.skipUnless(TORCH_AVAILABLE, "torch/torchvision not installed in this environment")
class CheckpointAndSmokeTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_weights(self):
        model = build_densenet121(8, dropout=0.0, pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pth"
            save_checkpoint_atomic({"model_state": model.state_dict(), "class_mapping": list(CLASSES)}, path)
            loaded = load_checkpoint(path)
        restored = build_densenet121(8, dropout=0.0, pretrained=False)
        restored.load_state_dict(loaded["model_state"])
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), restored.state_dict().items()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.equal(v1, v2), k1)
        self.assertEqual(loaded["class_mapping"], list(CLASSES))

    def test_one_cpu_training_epoch_is_finite(self):
        """Smoke test: a tiny synthetic epoch runs end to end and produces finite loss and probabilities."""
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = []
            for i in range(6):
                name = tmp / f"img_{i}.png"
                Image.fromarray(np.random.default_rng(i).integers(0, 255, (96, 96), dtype=np.uint8)).save(name)
                row = {"image_path": str(name)}
                for j, cls in enumerate(CLASSES):
                    row[cls] = int((i + j) % 3 == 0)
                rows.append(row)
            import pandas as pd
            manifest = pd.DataFrame(rows)
            for cls in CLASSES:  # guarantee at least one positive per class
                manifest.loc[0, cls] = 1

            model = build_densenet121(8, dropout=0.3, pretrained=False)
            set_unfreeze_level(model, 0)
            opt = build_optimizer(model, 1e-3, None, 0.0, "adamw")
            device = torch.device("cpu")
            loader = build_loader(ChestXrayDataset(manifest, CLASSES, build_transforms(PREPROCESSING)), 2,
                                  shuffle=True, sampler=None, num_workers=0, pin_memory=False, seed=0)
            scaler = make_grad_scaler(False)
            criterion = MultiLabelBCELoss(None, 0.0)

            loss, y_true, y_prob = train_one_epoch(model, loader, criterion, opt, scaler, device, use_amp=False)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(y_true.shape, (6, 8))
            self.assertTrue(((y_prob >= 0) & (y_prob <= 1)).all())

            val_loss, _, _ = collect_predictions(model, loader, device, use_amp=False, criterion=criterion)
            self.assertTrue(np.isfinite(val_loss))


if __name__ == "__main__":
    unittest.main()

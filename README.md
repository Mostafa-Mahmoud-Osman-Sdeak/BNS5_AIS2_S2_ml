# DiagnoTech Model 2 — Chest X-ray Multi-Label Classification

DenseNet121 (PyTorch, ImageNet transfer learning) for eight chest X-ray findings:
**Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, No Finding, Nodule, Pneumothorax.**

This is a research prototype for a graduation project. It is not a validated medical device.

---

## 1. Status: what has and has not been executed

| Component | Status |
|---|---|
| Data layer: label encoding, manifest audit, duplicate/leakage grouping, patient-level split, class statistics | **Unit-tested and integration-tested on synthetic data** (35 torch-free tests in total, all pass) |
| Metrics: per-class report, AUROC/AP, threshold tuning, experiment ranking | **Unit-tested** (all pass) |
| Model, transforms, checkpointing, training loop, resumption, inference, loader | **Syntax- and reference-checked only.** PyTorch was not available in the build environment, so these were never executed. Torch tests exist and skip automatically without torch. |
| Notebooks 01, 02, 03 | Structure validated (valid nbformat 4.5; every code cell compiles). **Not executed end-to-end.** |
| Dataset inspection (images, labels, class counts, patient overlap) | **Not performed.** It happens in Notebook 01 in Colab. |
| Training, evaluation metrics, deployment check | **Not performed.** No accuracy, AUROC, F1 or timing figures are reported anywhere in this package, because none were produced. |

Run `python -m unittest discover -s tests -v` in any environment with PyTorch to execute the torch tests as well.

---

## 2. Dataset findings and design consequences

The primary source is `NIH_Model2_7_Diseases.zip` (Google Drive file `14mTSe12PWrgRxMWFOme0johrmDgmBfaC`, about 2.36 GB, shared by link).

* **The name and the class list suggest an NIH ChestX-ray14 subset.** Seven diseases plus *No Finding* gives exactly the eight requested classes. That dataset labels each image with a `Finding Labels` string that can hold several findings, such as `Effusion|Mass`. It also provides `Image Index` and `Patient ID` columns.
* **Consequence: multi-label, not single-label.** A softmax classifier would discard co-occurring findings. The model has eight independent sigmoid outputs trained with binary cross-entropy. `No Finding` is kept as an ordinary output, and rows that mix it with a disease are flagged as contradictory and excluded, not guessed at.
* **Verified or not.** The archive contents could not be read from the build environment. The archive is 2.36 GB, the sandbox has no network access, and reading the file through Drive would load it into the conversation. Notebook 01 Section 3 checks the column names and stops if they differ. Section 4 measures how often images carry more than one finding, which settles the single-label versus multi-label question with real data.
* **Other Drive content was not used.** The shared Drive also holds a `data model 2` folder with class-named subfolders (`Normal`, `Pneumonia`, `Tuberculosis`, `cardiomegaly`, `lung cancer`) and a `data model 2.zip`. These do not match the eight-class specification and were not used. Confirm that the NIH archive is the intended primary dataset.

---

## 3. Directory layout

```
DiagnoTech_Model2/                 # code root (this zip, unzipped to /content in Colab)
├── README.md
├── requirements.txt
├── config/
│   ├── project_config.json        # classes, annotation schema, split, preprocessing, model, training
│   ├── experiments.json           # E0–E8 definitions (one documented change per step)
│   └── class_mapping.json         # fixed class-to-index order (part of the model contract)
├── src/diagnotech/
│   ├── config.py                  # config loading, storage paths, config snapshot
│   ├── data_manifest.py           # labels, manifests, image checks, leakage groups, split   [no torch]
│   ├── metrics.py                 # multi-label metrics, threshold tuning, ranking            [no torch]
│   ├── checkpointing.py           # atomic save/load, RNG capture
│   ├── model.py                   # DenseNet121 head, unfreezing levels, optimizer groups
│   ├── preprocessing.py           # transforms (shared by train/eval/inference), dataset, loaders
│   └── training.py                # staged training, early stopping, scheduler, resume, experiments
├── deployment/
│   ├── model_loader.py            # validated bundle loading (weights + class order + thresholds)
│   └── inference.py               # DiagnoTechPredictor: predict / predict_batch, CLI smoke test
├── notebooks/
│   ├── 01_Dataset_Preparation_Analysis.ipynb
│   ├── 02_DenseNet121_Training.ipynb
│   └── 03_Evaluation_Deployment.ipynb
└── tests/
    ├── test_data_manifest.py      # runs without torch
    ├── test_metrics.py            # runs without torch
    └── test_torch_components.py   # skipped automatically without torch
```

Storage root (Google Drive in Colab, `storage/` locally). Created automatically:

```
DiagnoTech_Model2/
├── data/           manifest_full.csv, manifest_{train,val,test}.csv, dataset_audit.json,
│                   split_summary.json, class_statistics.json, dataset_metadata.json
├── config/         snapshot of the configuration used by each run
├── checkpoints/<ID>/   best_model.pth, last_checkpoint.pth, training_history.csv, training_config.json
├── experiments/    experiment_results.csv, <ID>_val_per_class.csv
├── evaluation/     test reports, confusion matrices, ROC, error patterns, final_evaluation_summary.json
├── deployment/     best_model.pth, class_mapping.json, model_config.json
├── logs/           <ID>_train.log
└── reports/        EDA figures and training curves
```

---

## 4. Execution guide (Google Colab)

1. **Upload the code.** Upload `DiagnoTech_Model2_Professional.zip` to `/content` through the Colab file panel, then run:
   ```
   !unzip -q /content/DiagnoTech_Model2_Professional.zip -d /content
   ```
   This creates `/content/DiagnoTech_Model2/`. Set the runtime to **GPU** (Runtime → Change runtime type).
2. **Run Notebook 01** (`01_Dataset_Preparation_Analysis.ipynb`). It mounts Drive, downloads the archive to `/content/data_raw`, verifies labels and images, builds the patient-level split, and writes manifests to Drive. Read the printed audit and warnings before continuing. The run stops on any failed assertion.
3. **Run Notebook 02** (`02_DenseNet121_Training.ipynb`) once per experiment. Set `EXPERIMENT_ID` to `E0`, run, then `E1`, and so on through `E8`. Each run writes to its own checkpoint folder. Each completed run appends a row to `experiments/experiment_results.csv`. Run notebook 02 in the same session as notebook 01, because the extracted images live in `/content`.
4. **Run Notebook 03** (`03_Evaluation_Deployment.ipynb`). It selects the top-ranked eligible experiment, or the one you set in `SELECTED_EXPERIMENT`. It tunes thresholds on validation, evaluates once on test, writes the evaluation artifacts, exports `deployment/`, and checks that the deployment pipeline reproduces the training model's probabilities.

Runtimes have **not been benchmarked**. They depend on the Colab GPU, download speed, and the image count the archive contains. Notebook 01 is dominated by the archive download and the per-image checks. Notebook 02 is the bulk of the compute, with Phase B running up to 30 epochs per experiment. Plan the nine experiments as separate sessions, since Google Drive keeps the state between them. Check the per-epoch `epoch_seconds` column in `training_history.csv` to estimate the remaining budget.

### Restarting after a Colab disconnect
Reconnect, mount Drive, unzip the code, run notebook 01 Sections 1–2 (the archive is re-extracted), then run notebook 02 with the same `EXPERIMENT_ID` and `RESUME = True`. Training continues from the last completed epoch, restoring model, optimizer, scheduler, scaler, early-stopping counter, history and RNG state. A checkpoint created under a different configuration is refused, so settings cannot change silently mid-run.

---

## 5. Training design

**Phase A (feature extraction).** Backbone frozen, head trained at LR 1e-3, batch 16, up to 10 epochs.

**Phase B (fine-tuning).** Two stages, each with a fresh optimizer:
* Stage 1 unfreezes `denseblock4` and `norm5` (up to 15 epochs).
* Stage 2 additionally unfreezes `transition3` and `denseblock3` (up to 15 epochs).
* Head LR 5e-4, backbone LR 1e-4. Total budget 30 epochs.

Frozen BatchNorm layers stay in eval mode, so pretrained statistics are not overwritten.

**Architecture.** DenseNet121 features, global average pooling (built into torchvision's forward), `Dropout(0.3)`, `Linear(1024, 8)`, sigmoid per class.

**Augmentation (training split only).** Rotation ±5°, translation 4%, random resized crop (scale 0.9–1.0), brightness and contrast ±10%. **No horizontal or vertical flips**, because they alter cardiac and mediastinal laterality.

**Class imbalance.** Weights are computed from the training split only. `pos_weight = negatives / positives` per class (clipped at 50) for weighted BCE. The alternative is `WeightedRandomSampler`, with per-image weight equal to the mean inverse frequency of its positive labels (experiment E7).

**Model selection and stopping.** The best checkpoint is the epoch with the lowest validation loss, which is always unweighted and unsmoothed so it is comparable across experiments. Early stopping (patience 3 in Phase A, 4 per stage in Phase B) and ReduceLROnPlateau (mode min, factor 0.5, patience 2, min LR 1e-7) are active in E6–E8.

**Mixed precision.** AMP with gradient scaling on CUDA. The loss is computed in fp32. A non-finite loss halts training before a checkpoint is overwritten.

---

## 6. Experiment sequence

| ID | Change relative to predecessor |
|---|---|
| E0 | Baseline: head-only, Adam, no augmentation, no dropout, unweighted BCE, fixed budget |
| E1 | + conservative augmentation |
| E2 | + dropout 0.3 |
| E3 | + AdamW, weight decay 1e-4 |
| E4 | + staged fine-tuning with differential learning rates |
| E5 | + inverse-frequency `pos_weight` |
| E6 | + ReduceLROnPlateau and validation-driven early stopping |
| E7 | E6 with `WeightedRandomSampler` replacing `pos_weight` (comparison) |
| E8 | E6 + label smoothing 0.05 (optional comparison) |

**Ranking rule** (`metrics.rank_experiments`): an experiment is *eligible* only if no class AUROC drops more than 0.02 below E0. Among eligible experiments, the order is validation macro AUROC, then validation macro F1 at tuned thresholds, then lower validation loss. Macro F1 is shown but not used as the primary criterion, because its thresholds are tuned on the same validation split.

---

## 7. Evaluation protocol and leakage controls

* **Patient-level splitting.** Patients, not images, are separated. Patients who share an identical image file are merged into one group, which prevents a duplicated picture from leaking across splits. `verify_split` raises an error if any group or patient spans two splits.
* **Target split proportions** are 70 / 15 / 15 by image count, with a seeded, deterministic assignment. Rare-label groups are placed first so rare classes spread across splits. Warnings are printed when a validation or test class has fewer than 30 positives.
* **Test isolation.** Training (`run_experiment`) rejects any manifest other than `train` and `val`. The test manifest is loaded only in notebook 03.
* **Thresholds are tuned on validation only**, then applied unchanged to test.
* **Metrics for multi-label output.** Per-class precision, recall, specificity, F1, support, TP/FP/FN/TN, AUROC and average precision. Macro averages. Label accuracy (per label decision), exact-match accuracy (strict), confusion matrices per class, ROC curves, and the most frequent true-versus-predicted label-set error patterns.

---

## 8. Deployment and backend integration

Notebook 03 writes `deployment/best_model.pth`, `class_mapping.json`, and `model_config.json` (which holds the validation-tuned thresholds, preprocessing, dropout, and model version).

Integration:
```python
import sys
sys.path.insert(0, "/path/to/DiagnoTech_Model2/deployment")
from inference import DiagnoTechPredictor

predictor = DiagnoTechPredictor.from_directory("/path/to/deployment", device="cpu")  # load once at start-up
result = predictor.predict(image_bytes)  # bytes, a file path, or a PIL image
# result: predicted_labels, top_3, probabilities (all 8), thresholds, model_version, disclaimer
```

* `model_loader.load_model_bundle` refuses to load if the class order in `class_mapping.json`, `model_config.json` and the checkpoint disagree, or if any threshold is missing.
* Preprocessing is the same function used in training.
* The inference module adds `src/` to `sys.path` automatically when the project layout is preserved (`deployment/` and `src/` side by side). Otherwise install the `diagnotech` package from `src/`.
* CLI smoke test: `python deployment/inference.py --model-dir deployment/ chest.png`.

**ONNX export is not included.** Nothing in this package claims an ONNX deployment.

---

## 9. Deviations from the specification, and why

1. **Multi-label instead of single-label.** The specification permits this when annotations are multi-label, and the archive name indicates they are. Notebook 01 Section 4 measures it.
2. **Early stopping and LR scheduling start at E6.** The experiment sequence in the specification adds them there, so E0–E5 use fixed budgets. This keeps the comparison controlled, but it means E4 and E5 do not use validation-driven stopping. Use E6 or E8 as the candidate final configuration.
3. **Stage 2 continues from the current weights**, not from the best Stage 1 checkpoint. The best checkpoint across all stages is still kept.
4. **Optimizer state is rebuilt per stage**, because the trainable parameter set changes at each unfreezing step. This resets Adam moment estimates at stage boundaries.
5. **Reproducibility is seeded but not bit-exact.** cuDNN autotuning is enabled for speed.
6. **Duplicate detection is byte-level.** Re-encoded or near-duplicate images are not detected.
7. **Checkpoint files are large.** `last_checkpoint.pth` includes optimizer state and is written every epoch, which takes time on Drive.

---

## 10. Limitations

* **Label noise.** NIH-style labels were extracted automatically from radiology reports. Metrics measure agreement with these labels, not with independent clinical ground truth.
* **Minority classes.** Small positive counts in validation or test give wide per-class confidence intervals. Check the warnings from notebook 01.
* **No external validation.** Results apply to this archive only.
* **Not a medical device.** Outputs require review by a qualified clinician. Do not use for clinical decisions without independent regulatory and clinical validation.
* **Interpretation is not automated.** Notebook 03 produces the error analysis, but the written conclusions must come from reviewing those results.

---

## 11. Requirements

See `requirements.txt`. Python 3.10 or newer. PyTorch 2.1 or newer with a CUDA build for training. Colab's default runtime provides everything except `gdown`, which notebook 01 installs.

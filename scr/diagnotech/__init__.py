"""DiagnoTech Model 2: eight-output multi-label chest X-ray classifier (DenseNet121, PyTorch).

Submodules are imported explicitly to keep torch-free components usable on their own:
    diagnotech.config, diagnotech.data_manifest, diagnotech.metrics   (no torch)
    diagnotech.model, diagnotech.preprocessing, diagnotech.training   (torch + torchvision)
"""

__version__ = "1.0.0"

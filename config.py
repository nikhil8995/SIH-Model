import json
import os
from dataclasses import dataclass, asdict
from typing import List

PROJECT_ROOT = "/home/nikhil/Documents/python/projs/SIH"
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")


@dataclass
class DataConfig:
    classes_file: str = os.path.join(ARCHIVE_DIR, "classes.txt")
    train_dir: str = os.path.join(ARCHIVE_DIR, "dataset", "00 Test", "train")
    val_dir: str = os.path.join(ARCHIVE_DIR, "dataset", "00 Test", "val")
    test_dir: str = os.path.join(ARCHIVE_DIR, "dataset", "00 Test", "test")


@dataclass
class ModelConfig:
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    device: str = "cuda" if os.environ.get(
        "CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu"
    embedding_dim: int = 512
    prompt_templates: List[str] = None
    temperature: float = 0.01
    alpha_probe_blend: float = 0.7

    def __post_init__(self):
        if self.prompt_templates is None:
            self.prompt_templates = [
                "a photo of {}",
                "a photo of a {}",
                "a photo of a {} in a city",
                "a {} related civic issue",
                "an urban scene with {}",
                "{}",
            ]


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 10
    crop_padding_ratio: float = 0.12
    nms_iou_threshold: float = 0.6
    seed: int = 42


@dataclass
class ThresholdConfig:
    tau_high: float = 0.70
    tau_mid: float = 0.55
    tau_low: float = 0.45
    tau_eff: float = 0.65
    tau_submit: float = 0.60
    margin_delta: float = 0.12


@dataclass
class Paths:
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "ml", "artifacts")

    def ensure(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)


def read_classes(classes_file: str) -> List[str]:
    # Try provided path, then fallbacks
    candidate_paths = [
        classes_file,
        os.path.join(ARCHIVE_DIR, "classes.txt"),
        os.path.join(ARCHIVE_DIR, "data", "predefined_classes.txt"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
            return classes
    raise FileNotFoundError(
        f"Could not find classes file. Tried: {candidate_paths}")


def save_runtime_config(cfg_path: str, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, thr_cfg: ThresholdConfig, classes: List[str]):
    blob = {
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "thresholds": asdict(thr_cfg),
        "classes": classes,
    }
    with open(cfg_path, "w") as f:
        json.dump(blob, f, indent=2)

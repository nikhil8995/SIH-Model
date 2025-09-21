import os
from typing import List, Dict
import numpy as np
import torch
import open_clip
from PIL import Image

from .config import DataConfig, ModelConfig, TrainConfig, ThresholdConfig, Paths, read_classes


def load_clip(model_cfg: ModelConfig):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_cfg.clip_model_name,
        pretrained=model_cfg.clip_pretrained,
        device=model_cfg.device
    )
    tokenizer = open_clip.get_tokenizer(model_cfg.clip_model_name)
    model.eval()
    return model, preprocess, tokenizer


def build_prompt_texts(classes: List[str], templates: List[str]) -> List[str]:
    texts: List[str] = []
    for cls in classes:
        for t in templates:
            texts.append(t.format(cls.lower().replace('_', ' ')))
    return texts


def compute_text_features(model, tokenizer, texts: List[str], device: str) -> np.ndarray:
    with torch.no_grad():
        text_tokens = tokenizer(texts)
        text_tokens = text_tokens.to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)
    return text_features.float().cpu().numpy()


def compute_image_features(model, images: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
    return image_features


def cosine_similarity(image_features: torch.Tensor, text_features: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = image_features @ text_features.t()
    return sim / temperature

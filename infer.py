import os
import json
from typing import List, Tuple
import numpy as np
import torch
import joblib
from PIL import Image

from .config import Paths, ThresholdConfig
from .feature_extractor import load_clip, compute_text_features, build_prompt_texts, compute_image_features
from .dataset import YoloCropDataset
from .config import DataConfig, ModelConfig, TrainConfig, read_classes


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax with numerical stability"""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / (exp.sum(axis=-1, keepdims=True) + 1e-9)


def aggregate_crop_probs(probs_list: List[np.ndarray]) -> np.ndarray:
    """Aggregate probabilities across multiple crops by averaging"""
    if not probs_list:
        return np.array([])
    return np.mean(np.stack(probs_list, axis=0), axis=0)


def infer_on_split(split_dir: str):
    # Load configs
    paths = Paths()
    thr = ThresholdConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    classes = read_classes(data_cfg.classes_file)

    print(f"Inference for {len(classes)} classes: {classes}")

    # Load CLIP model
    model, preprocess, tokenizer = load_clip(model_cfg)

    # Load cached text features and prompt texts
    try:
        text_features = torch.tensor(np.load(os.path.join(
            paths.artifacts_dir, 'text_features.npy'))).to(model_cfg.device)
        prompt_texts_loaded = np.load(os.path.join(
            paths.artifacts_dir, 'prompt_texts.npy'), allow_pickle=True)

        # Rebuild prompt texts for consistency checking
        prompt_texts = build_prompt_texts(classes, model_cfg.prompt_templates)

        print(f"Loaded {len(prompt_texts)} prompts from artifacts")

    except FileNotFoundError as e:
        print(f"Error: Could not load artifacts - {e}")
        print("Please run train_probe.py first to generate the required artifacts")
        return

    # Load trained probe
    try:
        probe = joblib.load(os.path.join(
            paths.artifacts_dir, 'linear_probe.joblib'))
        print(f"Loaded probe with {probe.classes_.shape[0]} classes")
    except FileNotFoundError:
        print("Error: Could not load linear_probe.joblib")
        print("Please run train_probe.py first to train and save the probe")
        return

    # Create dataset
    ds = YoloCropDataset(
        split_dir, classes,
        crop_padding_ratio=train_cfg.crop_padding_ratio,
        nms_iou_threshold=train_cfg.nms_iou_threshold,
        preprocess=preprocess,
        return_image_level=True
    )

    results = []

    for batch, meta in torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False):
        batch = batch.squeeze(0)  # Remove batch dimension

        if batch.numel() == 0:  # Skip empty batches
            print(f"Warning: Empty batch for {meta['image_path'][0]}")
            continue

        # Compute image features for all crops
        img_feats = compute_image_features(
            model, batch, model_cfg.device)  # (K, D)

        # Zero-shot classification
        zs_logits = (img_feats @ text_features.t()) / model_cfg.temperature
        zs_probs = softmax(zs_logits.detach().cpu().numpy())  # (K, P)

        # Reduce prompt dimension to class dimension
        P = len(prompt_texts)  # Number of unique prompts
        C = len(classes)      # Number of unique classes

        # Verify consistency
        if P % C != 0:
            print(
                f"Error: Number of prompts ({P}) is not divisible by number of classes ({C})")
            print(
                "This indicates a mismatch between your current classes and saved artifacts")
            print("Please delete the artifacts directory and retrain the probe")
            return

        prompts_per_class = P // C
        print(f"Using {prompts_per_class} prompts per class")

        # Aggregate prompts for each class by taking max probability
        zs_probs_class = []
        for c_idx in range(C):
            start = c_idx * prompts_per_class
            end = (c_idx + 1) * prompts_per_class

            if end > P:
                print(
                    f"Error: Prompt indexing out of bounds for class {c_idx}")
                return

            # Take max over prompts for this class
            class_probs = zs_probs[:, start:end].max(axis=1)
            zs_probs_class.append(class_probs)

        zs_probs_class = np.stack(zs_probs_class, axis=1)  # (K, C)

        # Linear probe classification
        probe_probs = probe.predict_proba(img_feats.cpu().numpy())  # (K, C)

        # Verify shapes match
        if probe_probs.shape[1] != C:
            print(
                f"Error: Probe output classes ({probe_probs.shape[1]}) != expected classes ({C})")
            print("The probe was trained on a different number of classes")
            print("Please delete artifacts directory and retrain the probe")
            return

        # Debug information
        print(f"Processing {meta['image_path'][0]}:")
        print(f"  Crops: {batch.shape[0]}")
        print(f"  Zero-shot probs shape: {zs_probs_class.shape}")
        print(f"  Probe probs shape: {probe_probs.shape}")

        # Blend zero-shot and probe predictions
        blended = (model_cfg.alpha_probe_blend * probe_probs +
                   (1 - model_cfg.alpha_probe_blend) * zs_probs_class)

        # Aggregate across crops (average probabilities)
        crop_probs = blended
        img_probs = aggregate_crop_probs(
            [crop_probs[i] for i in range(crop_probs.shape[0])])

        # Make final prediction
        pred_idx = int(img_probs.argmax())
        confidence = float(img_probs.max())

        result = {
            'image_path': meta['image_path'][0],
            'pred_class': classes[pred_idx],
            'confidence': confidence,
            'all_probs': {classes[i]: float(img_probs[i]) for i in range(len(classes))}
        }

        results.append(result)
        print(
            f"  Prediction: {classes[pred_idx]} (confidence: {confidence:.3f})")

    # Output results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(json.dumps({'results': results}, indent=2))


def infer_single_image(image_path: str):
    """Inference on a single image file - useful for API"""
    # Load configs
    paths = Paths()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    classes = read_classes(data_cfg.classes_file)

    # Load CLIP model
    model, preprocess, tokenizer = load_clip(model_cfg)

    # Load cached artifacts
    text_features = torch.tensor(np.load(os.path.join(
        paths.artifacts_dir, 'text_features.npy'))).to(model_cfg.device)
    prompt_texts = build_prompt_texts(classes, model_cfg.prompt_templates)
    probe = joblib.load(os.path.join(
        paths.artifacts_dir, 'linear_probe.joblib'))

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)

    # Get image features
    img_feats = compute_image_features(model, image_tensor, model_cfg.device)

    # Zero-shot classification
    zs_logits = (img_feats @ text_features.t()) / model_cfg.temperature
    zs_probs = softmax(zs_logits.detach().cpu().numpy())

    # Reduce prompts to classes
    P = len(prompt_texts)
    C = len(classes)
    prompts_per_class = P // C

    zs_probs_class = []
    for c_idx in range(C):
        start = c_idx * prompts_per_class
        end = (c_idx + 1) * prompts_per_class
        zs_probs_class.append(zs_probs[:, start:end].max(axis=1))

    zs_probs_class = np.stack(zs_probs_class, axis=1)

    # Linear probe classification
    probe_probs = probe.predict_proba(img_feats.cpu().numpy())

    # Blend predictions
    blended = (model_cfg.alpha_probe_blend * probe_probs +
               (1 - model_cfg.alpha_probe_blend) * zs_probs_class)

    # Get final prediction
    pred_idx = int(blended[0].argmax())
    confidence = float(blended[0].max())

    return {
        'predicted_class': classes[pred_idx],
        'confidence': confidence,
        'all_probabilities': {classes[i]: float(blended[0][i]) for i in range(len(classes))}
    }


if __name__ == '__main__':
    data_cfg = DataConfig()
    infer_on_split(data_cfg.test_dir)

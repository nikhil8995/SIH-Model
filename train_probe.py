import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from .config import DataConfig, ModelConfig, TrainConfig, ThresholdConfig, Paths, read_classes, save_runtime_config
from .dataset import YoloCropDataset
from .feature_extractor import load_clip, build_prompt_texts, compute_text_features, compute_image_features


def extract_split_embeddings(split_dir: str, classes: list, model, preprocess, device: str, batch_size: int, crop_padding_ratio: float, nms_iou_threshold: float):
    ds = YoloCropDataset(split_dir, classes, crop_padding_ratio=crop_padding_ratio,
                         nms_iou_threshold=nms_iou_threshold, preprocess=preprocess, return_image_level=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    all_feats = []
    all_labels = []
    all_img_paths = []

    for batch, meta in dl:
        # variable number of crops, already stacked; shape (K, C, H, W)
        batch = batch.squeeze(0)
        if batch.numel() == 0:  # Skip empty batches
            continue

        feats = compute_image_features(model, batch, device)
        feats_np = feats.float().cpu().numpy()

        # Align labels length to features count
        labels_list = list(meta['target_classes'][0])
        K = feats_np.shape[0]

        if len(labels_list) < K:
            labels_list = labels_list + ([-1] * (K - len(labels_list)))
        elif len(labels_list) > K:
            labels_list = labels_list[:K]

        labels = np.asarray(labels_list, dtype=np.int64)
        all_feats.append(feats_np)
        all_labels.append(labels)
        all_img_paths.append(meta['image_path'][0])

    # Concatenate variable length by stacking and keeping per-crop labels
    feats_cat = np.concatenate(all_feats, axis=0) if all_feats else np.zeros(
        (0, 512), dtype=np.float32)
    labels_cat = np.concatenate(
        all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)

    return feats_cat, labels_cat, all_img_paths


def train_and_save_probe():
    # Diagnostic print statement
    print("--- EXECUTING train_probe.py (Fixed version) ---")

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    thr_cfg = ThresholdConfig()
    paths = Paths()
    paths.ensure()
    classes = read_classes(data_cfg.classes_file)

    print(f"Training probe for {len(classes)} classes: {classes}")

    model, preprocess, tokenizer = load_clip(model_cfg)

    # Text features for prompts (cache)
    prompt_texts = build_prompt_texts(classes, model_cfg.prompt_templates)
    text_features = compute_text_features(
        model, tokenizer, prompt_texts, model_cfg.device)

    print(f"Generated {len(prompt_texts)} prompts for {len(classes)} classes")

    # Save prompt texts and features
    np.save(os.path.join(paths.artifacts_dir, 'prompt_texts.npy'),
            np.array(prompt_texts, dtype=object))

    # Handle text_features whether it's a tensor or numpy array
    if isinstance(text_features, torch.Tensor):
        text_features_np = text_features.cpu().numpy()
    else:
        text_features_np = text_features

    np.save(os.path.join(paths.artifacts_dir,
            'text_features.npy'), text_features_np)

    # Extract embeddings
    print("Extracting training embeddings...")
    train_X, train_y, _ = extract_split_embeddings(
        data_cfg.train_dir, classes, model, preprocess,
        model_cfg.device, train_cfg.batch_size,
        train_cfg.crop_padding_ratio, train_cfg.nms_iou_threshold)

    print("Extracting validation embeddings...")
    val_X, val_y, _ = extract_split_embeddings(
        data_cfg.val_dir, classes, model, preprocess,
        model_cfg.device, train_cfg.batch_size,
        train_cfg.crop_padding_ratio, train_cfg.nms_iou_threshold)

    # Filter labeled crops (remove -1 labels)
    idx_train = train_y >= 0
    idx_val = val_y >= 0

    X = np.concatenate([train_X[idx_train], val_X[idx_val]], axis=0)
    y = np.concatenate([train_y[idx_train], val_y[idx_val]], axis=0)

    print(
        f"Training probe on {X.shape[0]} samples with {len(np.unique(y))} unique classes")
    print(f"Class distribution: {np.bincount(y)}")

    # Check if all classes are present in training data
    unique_classes = np.unique(y)
    missing_classes = set(range(len(classes))) - set(unique_classes)

    if missing_classes:
        print(f"Warning: Classes {missing_classes} have no training samples")
        print(f"Missing class names: {[classes[i] for i in missing_classes]}")
        print("Adding synthetic samples for missing classes...")

        # Add multiple synthetic samples for each missing class (minimum 5 for CV)
        # Use mean feature vector with different noise levels
        mean_features = X.mean(axis=0)
        synthetic_X = []
        synthetic_y = []

        samples_per_missing_class = 5  # Minimum for 3-fold CV

        for missing_class in missing_classes:
            for i in range(samples_per_missing_class):
                # Add different levels of noise for variety
                noise_scale = 0.01 + (i * 0.005)  # Varying noise levels
                synthetic_feat = mean_features + \
                    np.random.normal(0, noise_scale, mean_features.shape)
                synthetic_X.append(synthetic_feat)
                synthetic_y.append(missing_class)

        if synthetic_X:
            synthetic_X = np.array(synthetic_X)
            synthetic_y = np.array(synthetic_y)
            X = np.concatenate([X, synthetic_X], axis=0)
            y = np.concatenate([y, synthetic_y], axis=0)
            print(
                f"Added {len(synthetic_X)} synthetic samples ({samples_per_missing_class} per missing class)")

    # Train linear probe with calibration
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, n_jobs=-1)
    )

    # Check if we have enough samples per class for CV
    class_counts = np.bincount(y)
    min_samples = class_counts[class_counts > 0].min()

    if min_samples >= 3:
        # Use CalibratedClassifierCV for better probability estimates
        cal = CalibratedClassifierCV(clf, method='isotonic', cv=3)
    elif min_samples >= 2:
        # Use 2-fold CV
        print(
            f"Warning: Minimum class has {min_samples} samples, using 2-fold CV")
        cal = CalibratedClassifierCV(clf, method='isotonic', cv=2)
    else:
        # Skip calibration - use raw classifier
        print(
            f"Warning: Minimum class has {min_samples} samples, skipping calibration")
        cal = clf

    cal.fit(X, y)

    # Verify the probe learned all classes
    probe_classes = cal.classes_
    print(f"Probe learned {len(probe_classes)} classes: {probe_classes}")
    if len(probe_classes) != len(classes):
        print(
            f"ERROR: Probe has {len(probe_classes)} classes but expected {len(classes)}")
        return

    # Save the trained probe
    joblib.dump(cal, os.path.join(paths.artifacts_dir, 'linear_probe.joblib'))

    # Save runtime config
    save_runtime_config(
        os.path.join(paths.artifacts_dir, 'runtime_config.json'),
        data_cfg, model_cfg, train_cfg, thr_cfg, classes)

    print(f'Saved artifacts to {paths.artifacts_dir}')
    print(f'Probe trained for {len(classes)} classes')


if __name__ == '__main__':
    train_and_save_probe()

import os
import glob
from typing import List, Tuple, Optional, Dict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_yolo_boxes(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def _xywhn_to_xyxy_abs(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int((cx - w / 2.0) * W)
    y1 = int((cy - h / 2.0) * H)
    x2 = int((cx + w / 2.0) * W)
    y2 = int((cy + h / 2.0) * H)
    return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)


def _pad_box(x1, y1, x2, y2, W, H, pad_ratio: float) -> Tuple[int, int, int, int]:
    pw = int((x2 - x1 + 1) * pad_ratio)
    ph = int((y2 - y1 + 1) * pad_ratio)
    return max(0, x1 - pw), max(0, y1 - ph), min(W - 1, x2 + pw), min(H - 1, y2 + ph)


def _nms(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_th: float) -> List[int]:
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    idxs = scores_np.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.size == 1:
            break
        x1 = np.maximum(boxes_np[i, 0], boxes_np[idxs[1:], 0])
        y1 = np.maximum(boxes_np[i, 1], boxes_np[idxs[1:], 1])
        x2 = np.minimum(boxes_np[i, 2], boxes_np[idxs[1:], 2])
        y2 = np.minimum(boxes_np[i, 3], boxes_np[idxs[1:], 3])
        w = np.maximum(0, x2 - x1 + 1)
        h = np.maximum(0, y2 - y1 + 1)
        inter = w * h
        area_i = (boxes_np[i, 2] - boxes_np[i, 0] + 1) * \
            (boxes_np[i, 3] - boxes_np[i, 1] + 1)
        area_rest = (boxes_np[idxs[1:], 2] - boxes_np[idxs[1:], 0] + 1) * \
            (boxes_np[idxs[1:], 3] - boxes_np[idxs[1:], 1] + 1)
        iou = inter / (area_i + area_rest - inter + 1e-6)
        idxs = idxs[1:][iou < iou_th]
    return keep


class YoloCropDataset(Dataset):
    def __init__(self, split_dir: str, classes: List[str], crop_padding_ratio: float = 0.12,
                 nms_iou_threshold: float = 0.6, preprocess=None, return_image_level: bool = False):
        self.images_dir = os.path.join(split_dir, 'images')
        self.labels_dir = os.path.join(split_dir, 'labels')
        self.image_paths = sorted(
            glob.glob(os.path.join(self.images_dir, '*')))
        self.classes = classes
        self.crop_padding_ratio = crop_padding_ratio
        self.nms_iou_threshold = nms_iou_threshold
        self.preprocess = preprocess
        self.return_image_level = return_image_level

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        label_path = os.path.join(self.labels_dir, os.path.splitext(
            os.path.basename(img_path))[0] + '.txt')
        yolo_boxes = _read_yolo_boxes(label_path)
        crops: List[Image.Image] = []
        crop_boxes: List[Tuple[int, int, int, int]] = []
        crop_classes: List[int] = []
        for cls, cx, cy, w, h in yolo_boxes:
            x1, y1, x2, y2 = _xywhn_to_xyxy_abs(cx, cy, w, h, W, H)
            x1, y1, x2, y2 = _pad_box(
                x1, y1, x2, y2, W, H, self.crop_padding_ratio)
            crop = image.crop((x1, y1, x2+1, y2+1))
            crops.append(crop)
            crop_boxes.append((x1, y1, x2, y2))
            crop_classes.append(cls)
        # Deduplicate overlapping crops with simple NMS using area as score
        if len(crop_boxes) > 1:
            scores = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in crop_boxes]
            keep_idx = _nms(crop_boxes, scores, self.nms_iou_threshold)
            crops = [crops[i] for i in keep_idx]
            crop_boxes = [crop_boxes[i] for i in keep_idx]
            crop_classes = [crop_classes[i] for i in keep_idx]
        # Fallback to full image if no labels
        if not crops:
            crops = [image]
            crop_boxes = [(0, 0, W-1, H-1)]
            crop_classes = [-1]
        # Apply preprocess
        if self.preprocess is not None:
            crops = [self.preprocess(c) for c in crops]
            crops = [torch.tensor(np.array(c)) if not isinstance(
                c, torch.Tensor) else c for c in crops]
            crops = [c if c.ndim == 3 else c.permute(2, 0, 1) for c in crops]
        batch = torch.stack(crops, dim=0)
        meta: Dict[str, object] = {
            'image_path': img_path,
            'boxes': crop_boxes,
            'target_classes': crop_classes,
        }
        if self.return_image_level:
            # Image-level target if any crop has a labeled class
            img_cls = next((c for c in crop_classes if c >= 0), -1)
            meta['image_level_class'] = img_cls
        return batch, meta

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF

try:
    from torchvision.models import ResNet18_Weights  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    ResNet18_Weights = None  # type: ignore[assignment]

try:
    from torchvision.models import vit_b_16  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    vit_b_16 = None  # type: ignore[assignment]

try:
    from torchvision.models import ViT_B_16_Weights  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    ViT_B_16_Weights = None  # type: ignore[assignment]

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
except ImportError:  # pragma: no cover
    fasterrcnn_resnet50_fpn = None  # type: ignore[assignment]
    FasterRCNN_ResNet50_FPN_Weights = None  # type: ignore[assignment]

NUM_CLASSES = 3
TARGET_OBJECT_CLASS = 1
CLASS_NAMES = ["no_object", "object_of_interest", "another_object"]



@dataclass(frozen=True)
class Sample:
    path: Path
    label_name: str
    label_id: int


@dataclass(frozen=True)
class Entry:
    path: Path
    target: int


class SonarDataset(Dataset):
    def __init__(self, entries: Sequence[Entry], transform: transforms.Compose):
        self.entries = list(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, entry.target


class SonarPatchBagDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[Entry],
        image_size: int,
        patch_size: int,
        patch_stride: int,
        transform: transforms.Compose,
    ):
        if patch_size > image_size:
            raise ValueError("patch_size must be <= image_size")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        self.entries = list(entries)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.transform = transform
        self.coords = self._build_coords()

    def _build_coords(self) -> List[Tuple[int, int]]:
        coords: List[Tuple[int, int]] = []
        for y in range(0, self.image_size - self.patch_size + 1, self.patch_stride):
            for x in range(0, self.image_size - self.patch_size + 1, self.patch_stride):
                coords.append((x, y))
        if not coords:
            # ensure at least single crop covering image center
            offset = (self.image_size - self.patch_size) // 2
            coords.append((max(0, offset), max(0, offset)))
        return coords

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.path).convert("RGB").resize((self.image_size, self.image_size))
        patches: List[torch.Tensor] = []
        for x, y in self.coords:
            crop = image.crop((x, y, x + self.patch_size, y + self.patch_size))
            tensor = self.transform(crop) if self.transform else TF.to_tensor(crop)
            patches.append(tensor)
        bag = torch.stack(patches)
        return bag, entry.target


class PseudoDetectionDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[Entry],
        pseudo_boxes: Dict[str, List[Tuple[int, int, int, int, float]]],
        image_size: int,
        target_label: int,
        min_score: float = 0.0,
    ):
        self.entries = list(entries)
        self.image_size = image_size
        self.pseudo_boxes = pseudo_boxes
        self.target_label = target_label
        self.min_score = min_score

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.path).convert("RGB").resize((self.image_size, self.image_size))
        tensor = TF.to_tensor(image)
        key = str(entry.path)
        boxes_with_scores = self.pseudo_boxes.get(key, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        for x1, y1, x2, y2, score in boxes_with_scores:
            if score < self.min_score:
                continue
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(self.target_label)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return tensor, target


def generate_patch_coordinates(image_size: int, patch_size: int, patch_stride: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for y in range(0, image_size - patch_size + 1, patch_stride):
        for x in range(0, image_size - patch_size + 1, patch_stride):
            coords.append((x, y))
    if not coords:
        offset = (image_size - patch_size) // 2
        coords.append((max(0, offset), max(0, offset)))
    return coords


def extract_patch_bag(
    image: Image.Image,
    image_size: int,
    patch_size: int,
    patch_stride: int,
    transform: transforms.Compose,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    resized = image.resize((image_size, image_size))
    coords = generate_patch_coordinates(image_size, patch_size, patch_stride)
    patches: List[torch.Tensor] = []
    for x, y in coords:
        crop = resized.crop((x, y, x + patch_size, y + patch_size))
        tensor = transform(crop) if transform else TF.to_tensor(crop)
        patches.append(tensor)
    bag = torch.stack(patches)
    return bag, coords


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


class MILClassifier(nn.Module):
    def __init__(self, feature_extractor: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.instance_head = nn.Linear(feature_dim, num_classes)

    def forward(self, bags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # bags: (B, N, C, H, W)
        batch_size, num_instances, channels, height, width = bags.shape
        flat = bags.view(batch_size * num_instances, channels, height, width)
        features = self.feature_extractor(flat)
        if features.dim() == 4:
            features = torch.flatten(features, 1)
        instance_logits = self.instance_head(features)
        instance_logits = instance_logits.view(batch_size, num_instances, -1)
        bag_logits = instance_logits.max(dim=1).values
        return bag_logits, instance_logits


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(root: Path) -> List[Sample]:
    mapping = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 1,
        "object_c": 2,
    }
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    samples: List[Sample] = []
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label_name = class_dir.name
        if label_name not in mapping:
            continue
        label_id = mapping[label_name]
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() in exts:
                samples.append(Sample(path=fp, label_name=label_name, label_id=label_id))
    if not samples:
        raise RuntimeError(f"No samples found under {root}")
    return samples


def split_indices(samples: Sequence[Sample], val_size: float, test_size: float, seed: int) -> Dict[str, List[int]]:
    indices = np.arange(len(samples))
    labels = np.array([s.label_id for s in samples])

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    if val_size > 0:
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            stratify=labels[train_idx],
            random_state=seed,
        )
    else:
        val_idx = np.array([], dtype=int)
    return {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }


def build_entries(samples: Sequence[Sample], indices: Sequence[int]) -> List[Entry]:
    return [Entry(path=samples[i].path, target=samples[i].label_id) for i in indices]


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def make_patch_transform(patch_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_class_weights(entries: Sequence[Entry], num_classes: int) -> torch.Tensor:
    counts = Counter(e.target for e in entries)
    total = len(entries)
    weights = []
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count > 0:
            weights.append(total / (num_classes * count))
        else:
            weights.append(0.0)
    tensor = torch.tensor(weights, dtype=torch.float32)
    positive = tensor[tensor > 0]
    if positive.numel() > 0:
        tensor = tensor / positive.mean()
    return tensor


def prepare_sampler(entries: Sequence[Entry], num_classes: int, seed: int) -> Optional[WeightedRandomSampler]:
    if not entries:
        return None
    counts = Counter(e.target for e in entries)
    if len(counts) < 2:
        return None
    total = len(entries)
    weights = {
        cls: total / (num_classes * count)
        for cls, count in counts.items()
        if count > 0
    }
    sample_weights = [weights[e.target] for e in entries]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)


def build_model(num_classes: int, freeze_backbone: bool, backbone: str) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(pretrained=True)  # pragma: no cover - legacy
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if backbone == "vit_b_16":
        if vit_b_16 is None:
            raise ImportError("torchvision vit_b_16 model is unavailable; install torchvision>=0.13")
        weights = ViT_B_16_Weights.DEFAULT if 'ViT_B_16_Weights' in globals() and ViT_B_16_Weights is not None else None
        model = models.vit_b_16(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        head_module = model.heads
        replaced = False
        if hasattr(head_module, "head") and isinstance(head_module.head, nn.Linear):
            in_features = head_module.head.in_features
            head_module.head = nn.Linear(in_features, num_classes)
            replaced = True
        elif isinstance(head_module, nn.Sequential) and len(head_module) > 0 and isinstance(head_module[0], nn.Linear):
            in_features = head_module[0].in_features
            head_module[0] = nn.Linear(in_features, num_classes)
            replaced = True
        elif isinstance(head_module, nn.Linear):
            in_features = head_module.in_features
            model.heads = nn.Linear(in_features, num_classes)
            replaced = True
        if not replaced:
            raise RuntimeError("Unsupported ViT head structure; cannot replace classifier")
        if freeze_backbone:
            for param in model.heads.parameters():
                param.requires_grad = True
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_mil_model(num_classes: int, freeze_backbone: bool) -> MILClassifier:
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        base = models.resnet18(weights=weights)
    else:
        base = models.resnet18(pretrained=True)  # pragma: no cover - legacy
    if freeze_backbone:
        for name, param in base.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    feature_dim = base.fc.in_features
    base.fc = nn.Identity()
    return MILClassifier(base, feature_dim, num_classes)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict[str, float]] = []
    best_state = None
    best_metric = -float("inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total += images.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = running_corrects / max(total, 1)

        val_acc = 0.0
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            correct = 0
            val_total = 0
            loss_sum = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)
                    correct += torch.sum(preds == targets).item()
                    val_total += images.size(0)
                    loss_sum += loss.item() * images.size(0)
            val_acc = correct / max(val_total, 1)
            val_loss = loss_sum / max(val_total, 1)
            metric = val_acc
        else:
            metric = epoch_acc

        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.3f}, val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_mil_model(
    model: MILClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    class_weights: torch.Tensor,
) -> Tuple[MILClassifier, List[Dict[str, float]]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict[str, float]] = []
    best_state = None
    best_metric = -float("inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for bags, targets in train_loader:
            bags = bags.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(bags)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits, dim=1)
            running_loss += loss.item() * bags.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total += bags.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = running_corrects / max(total, 1)

        val_acc = 0.0
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            correct = 0
            val_total = 0
            loss_sum = 0.0
            with torch.no_grad():
                for bags, targets in val_loader:
                    bags = bags.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits, _ = model(bags)
                    loss = criterion(logits, targets)
                    preds = torch.argmax(logits, dim=1)
                    correct += torch.sum(preds == targets).item()
                    val_total += bags.size(0)
                    loss_sum += loss.item() * bags.size(0)
            val_acc = correct / max(val_total, 1)
            val_loss = loss_sum / max(val_total, 1)
            metric = val_acc
        else:
            metric = epoch_acc

        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"MIL Epoch {epoch}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.3f}, val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device, class_names: Sequence[str]) -> Dict[str, object]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits_list.append(outputs.cpu().numpy())
            labels_list.append(targets.cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist()
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy, "report": report, "confusion_matrix": matrix}


def heatmap_to_bbox(heatmap: np.ndarray, threshold: float) -> Optional[Tuple[int, int, int, int]]:
    mask = heatmap >= threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    heatmap_img = Image.fromarray(heatmap_uint8, mode="L").resize(image.size)
    colored = Image.new("RGBA", image.size, (255, 0, 0, 0))
    colored.putalpha(heatmap_img)
    base = image.convert("RGBA")
    overlay = Image.blend(base, colored, alpha=0.4)
    if bbox is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(bbox, outline=(0, 255, 0, 255), width=2)
    return overlay.convert("RGB")


def visualize_detections(
    model: nn.Module,
    dataset: SonarDataset,
    transform: transforms.Compose,
    device: torch.device,
    output_dir: Path,
    max_samples: int,
    target_class: int,
    threshold: float,
    image_size: int,
) -> List[Dict[str, object]]:
    model.eval()
    grad_cam = GradCAM(model, model.layer4)
    results: List[Dict[str, object]] = []

    count = 0
    for entry in dataset.entries:
        if entry.target != target_class:
            continue
        image = Image.open(entry.path).convert("RGB")
        resized = image.resize((image_size, image_size))
        tensor = transform(resized)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
        pred = int(torch.argmax(logits, dim=1).item())
        if pred != target_class:
            continue

        heatmap = grad_cam.generate(tensor, target_class)
        bbox = heatmap_to_bbox(heatmap, threshold)
        overlay = overlay_heatmap(resized, heatmap, bbox)
        filename = output_dir / f"{entry.path.stem}_gradcam.jpg"
        overlay.save(filename)

        results.append({
            "path": str(entry.path),
            "prediction": pred,
            "bbox": bbox,
            "output": str(filename),
        })
        count += 1
        if count >= max_samples:
            break

    grad_cam.remove_hooks()
    return results



def overlay_bbox(image: Image.Image, bbox: Optional[Tuple[int, int, int, int]], color=(0, 255, 0, 255)) -> Image.Image:
    overlay = image.convert("RGBA")
    if bbox is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(bbox, outline=color, width=2)
    return overlay.convert("RGB")


def evaluate_mil_classifier(model: MILClassifier, loader: DataLoader, device: torch.device, class_names: Sequence[str]) -> Dict[str, object]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for bags, targets in loader:
            bags = bags.to(device, non_blocking=True)
            logits, _ = model(bags)
            logits_list.append(logits.cpu().numpy())
            labels_list.append(targets.cpu().numpy())
    if logits_list:
        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        preds = np.argmax(logits, axis=1)
        report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
        matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist()
        accuracy = float((preds == labels).mean())
    else:
        logits = np.empty((0, len(class_names)), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        report = {}
        matrix = []
        accuracy = float('nan')
    return {"accuracy": accuracy, "report": report, "confusion_matrix": matrix}


def visualize_mil_detections(
    model: MILClassifier,
    entries: Sequence[Entry],
    device: torch.device,
    output_dir: Path,
    max_samples: int,
    target_class: int,
    image_size: int,
    patch_size: int,
    patch_stride: int,
    transform: transforms.Compose,
) -> List[Dict[str, object]]:
    model.eval()
    coords = generate_patch_coordinates(image_size, patch_size, patch_stride)
    results: List[Dict[str, object]] = []
    count = 0
    for entry in entries:
        if entry.target != target_class:
            continue
        image = Image.open(entry.path).convert("RGB")
        bag, _ = extract_patch_bag(image, image_size, patch_size, patch_stride, transform)
        bag = bag.unsqueeze(0).to(device)
        with torch.no_grad():
            bag_logits, instance_logits = model(bag)
        pred = int(torch.argmax(bag_logits, dim=1).item())
        if pred != target_class:
            continue
        class_scores = instance_logits[0, :, target_class]
        score, best_idx = torch.max(class_scores, dim=0)
        px, py = coords[int(best_idx.item())]
        bbox = (int(px), int(py), int(px + patch_size), int(py + patch_size))
        resized = image.resize((image_size, image_size))
        overlay = overlay_bbox(resized, bbox)
        filename = output_dir / f"{entry.path.stem}_mil.jpg"
        overlay.save(filename)
        results.append({
            "path": str(entry.path),
            "prediction": pred,
            "bbox": list(map(int, bbox)),
            "score": float(score.item()),
            "patch_index": int(best_idx.item()),
            "output": str(filename),
        })
        count += 1
        if count >= max_samples:
            break
    return results


def generate_gradcam_pseudo_boxes(
    model: nn.Module,
    dataset: SonarDataset,
    device: torch.device,
    target_class: int,
    threshold: float,
    image_size: int,
) -> Dict[str, List[Tuple[int, int, int, int, float]]]:
    if not hasattr(model, "layer4"):
        raise ValueError("Grad-CAM pseudo labels require a ResNet-like model with layer4 attribute")
    grad_cam = GradCAM(model, model.layer4)
    pseudo: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
    for entry in dataset.entries:
        image = Image.open(entry.path).convert("RGB")
        resized = image.resize((image_size, image_size))
        if dataset.transform:
            tensor = dataset.transform(resized)
        else:
            tensor = TF.to_tensor(resized)
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        pred = int(torch.argmax(logits, dim=1).item())
        key = str(entry.path)
        pseudo[key] = []
        if pred != target_class:
            continue
        heatmap = grad_cam.generate(tensor, target_class)
        bbox = heatmap_to_bbox(heatmap, threshold)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        region = heatmap[y1 : y2 + 1, x1 : x2 + 1]
        score = float(region.max()) if region.size else 0.0
        pseudo[key].append((int(x1), int(y1), int(x2), int(y2), score))
    grad_cam.remove_hooks()
    return pseudo


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_detection_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Module:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.1)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(1, batches)
        print(f"Detector Epoch {epoch}: loss={avg_loss:.4f}")
    return model


def run_detection_inference(
    model: nn.Module,
    entries: Sequence[Entry],
    device: torch.device,
    image_size: int,
    score_threshold: float,
    target_class: int,
    max_samples: int,
    output_dir: Path,
) -> List[Dict[str, object]]:
    model.eval()
    results: List[Dict[str, object]] = []
    count = 0
    for entry in entries:
        image = Image.open(entry.path).convert("RGB")
        resized = image.resize((image_size, image_size))
        tensor = TF.to_tensor(resized).to(device)
        with torch.no_grad():
            outputs = model([tensor])
        if not outputs:
            continue
        output = outputs[0]
        boxes = output.get("boxes")
        labels = output.get("labels")
        scores = output.get("scores")
        if boxes is None or labels is None or scores is None:
            continue
        keep = (labels == target_class) & (scores >= score_threshold)
        if keep.sum().item() == 0:
            continue
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        best_idx = torch.argmax(filtered_scores)
        bbox = filtered_boxes[best_idx].cpu().tolist()
        score = float(filtered_scores[best_idx].cpu().item())
        bbox_int = [int(round(v)) for v in bbox]
        overlay = overlay_bbox(resized, tuple(bbox_int))
        filename = output_dir / f"{entry.path.stem}_detector.jpg"
        overlay.save(filename)
        results.append({
            "path": str(entry.path),
            "bbox": bbox_int,
            "score": score,
            "output": str(filename),
        })
        count += 1
        if count >= max_samples:
            break
    return results


class ViTAttentionRollout:
    def __init__(self, model: nn.Module, discard_ratio: float = 0.0):
        self.model = model
        self.discard_ratio = discard_ratio
        encoder = getattr(model, "encoder", None)
        if encoder is None or not hasattr(encoder, "layers"):
            raise ValueError("Model does not expose encoder layers for attention rollout")
        self.attentions: List[torch.Tensor] = []
        self.handles = [layer.attention.attn_drop.register_forward_hook(self._hook) for layer in encoder.layers]

    def _hook(self, module, inputs, output):
        self.attentions.append(output.detach())

    def clear(self) -> None:
        self.attentions.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _rollout(self) -> torch.Tensor:
        if not self.attentions:
            raise RuntimeError("Attention hooks did not capture any tensors")
        result: Optional[torch.Tensor] = None
        for attn in self.attentions:
            attn = attn.mean(dim=1)
            if self.discard_ratio > 0:
                flat = attn.view(attn.size(0), -1)
                num_discard = int(flat.size(1) * self.discard_ratio)
                if num_discard > 0:
                    indices = torch.argsort(flat, dim=1)
                    flat.scatter_(1, indices[:, :num_discard], 0.0)
                attn = flat.view_as(attn)
            identity = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0)
            attn = attn + identity
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = attn if result is None else torch.bmm(result, attn)
        return result

    def generate(self, inputs: torch.Tensor) -> torch.Tensor:
        self.clear()
        _ = self.model(inputs)
        rollout = self._rollout()
        return rollout


def visualize_vit_attention(
    model: nn.Module,
    dataset: SonarDataset,
    device: torch.device,
    output_dir: Path,
    max_samples: int,
    target_class: int,
    image_size: int,
    threshold: float,
    discard_ratio: float,
) -> List[Dict[str, object]]:
    rollout = ViTAttentionRollout(model, discard_ratio)
    results: List[Dict[str, object]] = []
    count = 0
    for entry in dataset.entries:
        if entry.target != target_class:
            continue
        image = Image.open(entry.path).convert("RGB")
        resized = image.resize((image_size, image_size))
        if dataset.transform:
            tensor = dataset.transform(resized)
        else:
            tensor = TF.to_tensor(resized)
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        pred = int(torch.argmax(logits, dim=1).item())
        if pred != target_class:
            continue
        attn = rollout.generate(tensor)
        if attn.dim() != 3:
            continue
        attn_map = attn[:, 0, 1:]
        grid = int(math.sqrt(attn_map.size(-1)))
        attn_map = attn_map.view(1, 1, grid, grid)
        attn_map = F.interpolate(attn_map, size=(image_size, image_size), mode="bilinear", align_corners=False)
        heatmap = attn_map.squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        bbox_tuple = heatmap_to_bbox(heatmap, threshold)
        overlay = overlay_heatmap(resized, heatmap, bbox_tuple)
        filename = output_dir / f"{entry.path.stem}_vit.jpg"
        overlay.save(filename)
        results.append({
            "path": str(entry.path),
            "prediction": pred,
            "bbox": None if bbox_tuple is None else list(map(int, bbox_tuple)),
            "output": str(filename),
        })
        count += 1
        if count >= max_samples:
            break
    rollout.remove()
    return results









def create_classification_dataloaders(
    train_entries: Sequence[Entry],
    val_entries: Sequence[Entry],
    test_entries: Sequence[Entry],
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    args: argparse.Namespace,
    sampler: Optional[WeightedRandomSampler],
) -> Tuple[SonarDataset, Optional[SonarDataset], SonarDataset, DataLoader, Optional[DataLoader], DataLoader]:
    train_dataset = SonarDataset(train_entries, train_transform)
    val_dataset = SonarDataset(val_entries, eval_transform) if val_entries else None
    test_dataset = SonarDataset(test_entries, eval_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_mil_dataloaders(
    train_entries: Sequence[Entry],
    val_entries: Sequence[Entry],
    test_entries: Sequence[Entry],
    args: argparse.Namespace,
    patch_transform: transforms.Compose,
) -> Tuple[SonarPatchBagDataset, Optional[SonarPatchBagDataset], SonarPatchBagDataset, DataLoader, Optional[DataLoader], DataLoader]:
    train_dataset = SonarPatchBagDataset(
        train_entries,
        args.image_size,
        args.mil_patch_size,
        args.mil_patch_stride,
        patch_transform,
    )
    val_dataset = (
        SonarPatchBagDataset(
            val_entries,
            args.image_size,
            args.mil_patch_size,
            args.mil_patch_stride,
            patch_transform,
        )
        if val_entries
        else None
    )
    test_dataset = SonarPatchBagDataset(
        test_entries,
        args.image_size,
        args.mil_patch_size,
        args.mil_patch_stride,
        patch_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def load_or_train_classifier(
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, List[Dict[str, float]], bool]:
    model = build_model(NUM_CLASSES, args.freeze_backbone, args.backbone)
    history: List[Dict[str, float]] = []
    loaded = False
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        state_dict = state.get("state_dict") if isinstance(state, dict) and "state_dict" in state else state
        if not isinstance(state_dict, dict):
            raise TypeError("Loaded weights must be a state dict or contain a 'state_dict' key")
        model.load_state_dict(state_dict)
        loaded = True
    else:
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.use_amp and device.type == "cuda",
            class_weights,
        )
    return model, history, loaded


def run_gradcam_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    train_entries: Sequence[Entry],
    val_entries: Sequence[Entry],
    test_entries: Sequence[Entry],
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    sampler: Optional[WeightedRandomSampler],
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, Dict[str, object]]:
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_classification_dataloaders(
        train_entries,
        val_entries,
        test_entries,
        train_transform,
        eval_transform,
        args,
        sampler,
    )
    model, history, loaded = load_or_train_classifier(args, device, train_loader, val_loader, class_weights)
    classification_metrics = evaluate_classifier(model, test_loader, device, CLASS_NAMES)
    if not hasattr(model, "layer4"):
        raise ValueError("Grad-CAM pipeline requires a ResNet-like backbone exposing layer4")
    detections = visualize_detections(
        model,
        test_dataset,
        eval_transform,
        device,
        Path(args.output_dir),
        args.max_visualizations,
        target_class=TARGET_OBJECT_CLASS,
        threshold=args.heatmap_threshold,
        image_size=args.image_size,
    )
    summary: Dict[str, object] = {
        "pipeline": "gradcam",
        "history": history,
        "classification": classification_metrics,
        "detections": detections,
        "backbone": args.backbone,
        "classifier_trained": not loaded,
    }
    return model, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sonar classifiers and derive weakly supervised detections using multiple pipelines.")
    parser.add_argument("--data-dir", type=str, default="datasets/data_sonar")
    parser.add_argument("--output-dir", type=str, default="runs_detection")
    parser.add_argument("--weights", type=str, default="", help="Optional path to pretrained classifier weights.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-visualizations", type=int, default=20)
    parser.add_argument("--heatmap-threshold", type=float, default=0.5)
    parser.add_argument("--localization-mode", type=str, choices=["gradcam", "mil", "self_train", "attention"], default="gradcam")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone for classifier-based pipelines (resnet18 or vit_b_16).")
    parser.add_argument("--mil-patch-size", type=int, default=64)
    parser.add_argument("--mil-patch-stride", type=int, default=32)
    parser.add_argument("--detector-epochs", type=int, default=5)
    parser.add_argument("--detector-batch-size", type=int, default=2)
    parser.add_argument("--detector-lr", type=float, default=1e-3)
    parser.add_argument("--detector-score-threshold", type=float, default=0.4)
    parser.add_argument("--attention-discard-ratio", type=float, default=0.0)
    parser.add_argument("--pseudo-score-threshold", type=float, default=0.0, help="Ignore pseudo boxes with scores below this threshold during detector training.")
    return parser.parse_args()
def run_attention_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    train_entries: Sequence[Entry],
    val_entries: Sequence[Entry],
    test_entries: Sequence[Entry],
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    sampler: Optional[WeightedRandomSampler],
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, Dict[str, object]]:
    if args.backbone.lower() != "vit_b_16":
        raise ValueError("Attention pipeline expects --backbone vit_b_16")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_classification_dataloaders(
        train_entries,
        val_entries,
        test_entries,
        train_transform,
        eval_transform,
        args,
        sampler,
    )
    model, history, loaded = load_or_train_classifier(args, device, train_loader, val_loader, class_weights)
    classification_metrics = evaluate_classifier(model, test_loader, device, CLASS_NAMES)
    detections = visualize_vit_attention(
        model,
        test_dataset,
        device,
        Path(args.output_dir),
        args.max_visualizations,
        target_class=TARGET_OBJECT_CLASS,
        image_size=args.image_size,
        threshold=args.heatmap_threshold,
        discard_ratio=args.attention_discard_ratio,
    )
    summary: Dict[str, object] = {
        "pipeline": "attention",
        "history": history,
        "classification": classification_metrics,
        "detections": detections,
        "backbone": args.backbone,
        "classifier_trained": not loaded,
    }
    return model, summary


def run_self_training_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    train_entries: Sequence[Entry],
    val_entries: Sequence[Entry],
    test_entries: Sequence[Entry],
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    sampler: Optional[WeightedRandomSampler],
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, Dict[str, object]]:
    if fasterrcnn_resnet50_fpn is None:
        raise ImportError("torchvision detection module not available; cannot run self-training pipeline")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_classification_dataloaders(
        train_entries,
        val_entries,
        test_entries,
        train_transform,
        eval_transform,
        args,
        sampler,
    )
    model, history, loaded = load_or_train_classifier(args, device, train_loader, val_loader, class_weights)
    classification_metrics = evaluate_classifier(model, test_loader, device, CLASS_NAMES)

    if not hasattr(model, "layer4"):
        raise ValueError("Self-training pipeline requires a ResNet-like backbone exposing layer4")

    pseudo_dataset = SonarDataset(train_entries, eval_transform)
    pseudo_boxes = generate_gradcam_pseudo_boxes(
        model,
        pseudo_dataset,
        device,
        target_class=TARGET_OBJECT_CLASS,
        threshold=args.heatmap_threshold,
        image_size=args.image_size,
    )
    filtered_counts = sum(
        sum(1 for _, _, _, _, score in boxes if score >= args.pseudo_score_threshold)
        for boxes in pseudo_boxes.values()
    )
    detection_results: List[Dict[str, object]] = []
    detector_summary: Dict[str, object] = {
        "epochs": args.detector_epochs,
        "lr": args.detector_lr,
        "pseudo_boxes": int(filtered_counts),
        "pseudo_threshold": args.pseudo_score_threshold,
    }

    if filtered_counts > 0:
        detection_dataset = PseudoDetectionDataset(
            train_entries,
            pseudo_boxes,
            args.image_size,
            TARGET_OBJECT_CLASS,
            min_score=args.pseudo_score_threshold,
        )
        detection_loader = DataLoader(
            detection_dataset,
            batch_size=args.detector_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=detection_collate,
        )
        detector_weights = None
        if 'FasterRCNN_ResNet50_FPN_Weights' in globals() and FasterRCNN_ResNet50_FPN_Weights is not None:
            detector_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn(weights=detector_weights, num_classes=2)
        detector.transform.min_size = [args.image_size]
        detector.transform.max_size = args.image_size
        detector = train_detection_model(detector, detection_loader, device, args.detector_epochs, args.detector_lr)
        detection_results = run_detection_inference(
            detector,
            test_entries,
            device,
            image_size=args.image_size,
            score_threshold=args.detector_score_threshold,
            target_class=TARGET_OBJECT_CLASS,
            max_samples=args.max_visualizations,
            output_dir=Path(args.output_dir),
        )
        detector_summary["trained"] = True
        detector_summary["score_threshold"] = args.detector_score_threshold
        torch.save(detector.state_dict(), Path(args.output_dir) / "detector.pt")
    else:
        detector_summary["trained"] = False
        detector_summary["reason"] = "No pseudo boxes passed the score threshold"

    summary: Dict[str, object] = {
        "pipeline": "self_train",
        "history": history,
        "classification": classification_metrics,
        "detections": detection_results,
        "backbone": args.backbone,
        "classifier_trained": not loaded,
        "detector": detector_summary,
    }
    return model, summary


def run_mil_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    train_entries: Sequence[Entry],
    val_entries: Sequence[Entry],
    test_entries: Sequence[Entry],
    class_weights: torch.Tensor,
) -> Tuple[MILClassifier, Dict[str, object]]:
    patch_transform = make_patch_transform(args.mil_patch_size)
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_mil_dataloaders(
        train_entries,
        val_entries,
        test_entries,
        args,
        patch_transform,
    )
    model = build_mil_model(NUM_CLASSES, args.freeze_backbone)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        history: List[Dict[str, float]] = []
        trained = False
    else:
        model, history = train_mil_model(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.use_amp and device.type == "cuda",
            class_weights,
        )
        trained = True
    classification_metrics = evaluate_mil_classifier(model, test_loader, device, CLASS_NAMES)
    detections = visualize_mil_detections(
        model,
        test_entries,
        device,
        Path(args.output_dir),
        args.max_visualizations,
        TARGET_OBJECT_CLASS,
        args.image_size,
        args.mil_patch_size,
        args.mil_patch_stride,
        patch_transform,
    )
    summary: Dict[str, object] = {
        "pipeline": "mil",
        "history": history,
        "classification": classification_metrics,
        "detections": detections,
        "classifier_trained": trained,
        "mil": {
            "patch_size": args.mil_patch_size,
            "patch_stride": args.mil_patch_stride,
            "num_patches": len(generate_patch_coordinates(args.image_size, args.mil_patch_size, args.mil_patch_stride)),
        },
    }
    return model, summary





def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    samples = collect_samples(data_dir)
    split_idx = split_indices(samples, args.val_size, args.test_size, args.seed)

    train_entries = build_entries(samples, split_idx["train"])
    val_entries = build_entries(samples, split_idx["val"])
    test_entries = build_entries(samples, split_idx["test"])

    train_transform, eval_transform = make_transforms(args.image_size)

    sampler = prepare_sampler(train_entries, num_classes=NUM_CLASSES, seed=args.seed)
    class_weights = compute_class_weights(train_entries, num_classes=NUM_CLASSES)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.localization_mode == "mil":
        model, summary = run_mil_pipeline(
            args,
            device,
            train_entries,
            val_entries,
            test_entries,
            class_weights,
        )
    elif args.localization_mode == "attention":
        model, summary = run_attention_pipeline(
            args,
            device,
            train_entries,
            val_entries,
            test_entries,
            train_transform,
            eval_transform,
            sampler,
            class_weights,
        )
    elif args.localization_mode == "self_train":
        model, summary = run_self_training_pipeline(
            args,
            device,
            train_entries,
            val_entries,
            test_entries,
            train_transform,
            eval_transform,
            sampler,
            class_weights,
        )
    else:
        model, summary = run_gradcam_pipeline(
            args,
            device,
            train_entries,
            val_entries,
            test_entries,
            train_transform,
            eval_transform,
            sampler,
            class_weights,
        )

    classifier_trained = bool(summary.get("classifier_trained", False))
    if classifier_trained and not args.weights:
        weight_name = "mil_classifier.pt" if args.localization_mode == "mil" else "classifier.pt"
        torch.save(model.state_dict(), output_dir / weight_name)

    metadata = summary.setdefault("metadata", {})
    metadata.update(
        {
            "train_samples": len(train_entries),
            "val_samples": len(val_entries),
            "test_samples": len(test_entries),
            "device": str(device),
            "image_size": args.image_size,
            "localization_mode": args.localization_mode,
        }
    )

    if "classification" in summary:
        print("\nClassification metrics:")
        print(json.dumps(summary["classification"], indent=2))

    if "detector" in summary:
        print("\nDetector summary:")
        print(json.dumps(summary["detector"], indent=2))

    summary_path = output_dir / "detection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")




if __name__ == "__main__":
    main()

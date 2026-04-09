import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

try:
    from torchvision.models import ResNet18_Weights  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    ResNet18_Weights = None  # type: ignore[assignment]

import matplotlib.pyplot as plt 
# --- add near other imports ---
from numpy.typing import NDArray

# --- add below collect_penultimate_features_* helpers ---
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def fit_ooi_prototype(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute normalized OOI (class 1) prototype in feature space."""
    pos = features[labels == 1]
    if pos.size == 0:
        raise RuntimeError("No positive samples to build the OOI prototype.")
    mu = pos.mean(axis=0, keepdims=True)
    return l2_normalize(mu)[0]

def cosine_similarity_to(mu: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row in feats and prototype mu."""
    feats_n = l2_normalize(feats)
    mu_n = mu / max(np.linalg.norm(mu), 1e-12)
    return feats_n @ mu_n

def calibrate_tau_sim(sim_pos: np.ndarray, keep_tpr: float = 0.95) -> float:
    """
    Given cosine similarities on validation positives, set tau_sim so that 
    at least `keep_tpr` of positives pass the gate.
    """
    q = (1.0 - keep_tpr) * 100.0
    return float(np.percentile(sim_pos, q))

def cosine_logits(features: np.ndarray, w: np.ndarray, t: float = 16.0) -> np.ndarray:
    """
    Post-hoc cosine classifier logits for 2 classes. 
    w: (2, D) from your trained linear layer; features: (N, D)
    """
    W = l2_normalize(w)
    X = l2_normalize(features)
    return t * (X @ W.T)  # temperature-scaled cosine logits


@dataclass(frozen=True)
class Sample:
    path: Path
    original_label: str
    binary_label: int  # 1 = object of interest, 0 = background


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

    def extend(self, new_entries: Sequence[Entry]) -> None:
        if not new_entries:
            return
        self.entries.extend(new_entries)


class ImageOnlyDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform: transforms.Compose, return_paths: bool = False):
        self.paths = list(paths)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_paths:
            return image, path
        return image


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(root: Path) -> List[Sample]:
    mapping = {
        "object_a": 1,
        "object_b": 1,
        "no_object": 0,
    }
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    samples: List[Sample] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in mapping and class_dir.name != "object_c":
            continue
        if class_dir.name == "object_c":
            continue
        binary = mapping[class_dir.name]
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() in exts:
                samples.append(Sample(path=fp, original_label=class_dir.name, binary_label=binary))
    if not samples:
        raise RuntimeError(f"No samples found in {root}")
    return samples


def collect_ood_samples(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ood_dir = root / "object_c"
    if not ood_dir.exists():
        return []
    paths = [fp for fp in sorted(ood_dir.iterdir()) if fp.suffix.lower() in exts]
    return paths


def split_indices(samples: Sequence[Sample], val_size: float, test_size: float, seed: int) -> Dict[str, List[int]]:
    indices = np.arange(len(samples))
    labels = np.array([s.binary_label for s in samples])

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
    return [Entry(path=samples[i].path, target=samples[i].binary_label) for i in indices]


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def prepare_sampler(entries: Sequence[Entry], num_classes: int, seed: int) -> Optional[WeightedRandomSampler]:
    if not entries:
        return None
    counts = Counter(e.target for e in entries)
    if len(counts) < 2:
        return None
    total = len(entries)
    class_weights = {
        label: total / (num_classes * count)
        for label, count in counts.items()
        if count > 0
    }
    sample_weights = [class_weights[e.target] for e in entries]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)


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


def build_model(freeze_backbone: bool) -> nn.Module:
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(pretrained=True)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


def train_model(
    model: nn.Module,
    train_dataset: SonarDataset,
    val_dataset: Optional[SonarDataset],
    device: torch.device,
    args: argparse.Namespace,
    ood_paths: Sequence[Path],
    eval_transform: transforms.Compose,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, object]]:
    epochs = args.epochs
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    val_loader: Optional[DataLoader]
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_loader = None

    oe_weight = args.outlier_exposure_weight
    oe_loader: Optional[DataLoader]
    if oe_weight > 0.0 and ood_paths:
        oe_dataset = ImageOnlyDataset(ood_paths, eval_transform)
        oe_loader = DataLoader(
            oe_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        oe_loader = None

    history: List[Dict[str, float]] = []
    initial_train_size = len(train_dataset)
    hard_negative_records: List[Dict[str, object]] = []
    oe_epoch_losses: List[float] = []
    hard_negative_paths: Set[str] = set()
    total_hard_negatives = 0

    best_state = None
    best_metric = -float("inf")

    for epoch in range(1, epochs + 1):
        sampler = prepare_sampler(train_dataset.entries, num_classes=2, seed=args.seed + epoch)
        class_weights = compute_class_weights(train_dataset.entries, num_classes=2)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=args.label_smoothing,
        )

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        oe_running_loss = 0.0
        oe_steps = 0
        oe_iterator = iter(oe_loader) if oe_loader is not None else None

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
                if oe_iterator is not None:
                    try:
                        oe_images = next(oe_iterator)
                    except StopIteration:
                        oe_iterator = iter(oe_loader)
                        oe_images = next(oe_iterator)
                    oe_images = oe_images.to(device, non_blocking=True)
                    oe_logits = model(oe_images)
                    uniform = torch.full_like(oe_logits, 1.0 / oe_logits.size(1))
                    oe_loss = F.kl_div(
                        F.log_softmax(oe_logits, dim=1),
                        uniform,
                        reduction="batchmean",
                    )
                    loss = loss + oe_weight * oe_loss
                    oe_running_loss += oe_loss.detach().item()
                    oe_steps += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.detach().item() * images.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total += images.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = running_corrects / max(total, 1)
        oe_avg_loss = oe_running_loss / max(oe_steps, 1) if oe_steps > 0 else 0.0
        if oe_steps > 0:
            oe_epoch_losses.append(oe_avg_loss)

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
                    val_loss_batch = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)
                    correct += torch.sum(preds == targets).item()
                    val_total += images.size(0)
                    loss_sum += val_loss_batch.item() * images.size(0)
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
            "dataset_size": len(train_dataset),
            "oe_loss": oe_avg_loss,
        })

        hard_added = 0
        hard_details: List[Dict[str, object]] = []
        if (
            args.hard_negative_topk > 0
            and epoch >= args.hard_negative_start_epoch
            and (args.hard_negative_max == 0 or total_hard_negatives < args.hard_negative_max)
        ):
            remaining = args.hard_negative_max - total_hard_negatives if args.hard_negative_max > 0 else args.hard_negative_topk
            remaining = max(0, remaining)
            request_k = min(args.hard_negative_topk, remaining) if remaining > 0 else 0
            if request_k > 0:
                new_entries, details = mine_hard_negative_entries(
                    model,
                    ood_paths,
                    eval_transform,
                    device,
                    request_k,
                    args.hard_negative_threshold,
                    hard_negative_paths,
                    args.batch_size,
                    args.num_workers,
                )
                filtered_entries: List[Entry] = []
                filtered_details: List[Dict[str, object]] = []
                for entry, detail in zip(new_entries, details):
                    path_str = detail["path"]
                    if path_str in hard_negative_paths:
                        continue
                    hard_negative_paths.add(path_str)
                    filtered_entries.append(entry)
                    filtered_details.append(detail)
                    if args.hard_negative_max and total_hard_negatives + len(filtered_entries) >= args.hard_negative_max:
                        break
                if filtered_entries:
                    train_dataset.extend(filtered_entries)
                    hard_added = len(filtered_entries)
                    total_hard_negatives += hard_added
                    hard_details = filtered_details
        hard_negative_records.append({
            "epoch": epoch,
            "added": hard_added,
            "details": hard_details,
            "dataset_size": len(train_dataset),
        })

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.3f}, "
            f"val_acc={val_acc:.3f}, hard_neg_added={hard_added}, oe_loss={oe_avg_loss:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    extras = {
        "hard_negative": {
            "enabled": args.hard_negative_topk > 0,
            "top_k": args.hard_negative_topk,
            "threshold": args.hard_negative_threshold,
            "start_epoch": args.hard_negative_start_epoch,
            "max_added": args.hard_negative_max,
            "total_added": total_hard_negatives,
            "per_epoch": hard_negative_records,
        },
        "outlier_exposure": {
            "weight": oe_weight,
            "avg_epoch_loss": sum(oe_epoch_losses) / len(oe_epoch_losses) if oe_epoch_losses else 0.0,
        },
        "label_smoothing": args.label_smoothing,
    }
    extras["epochs"] = epochs
    extras["initial_dataset_size"] = initial_train_size
    extras["final_dataset_size"] = len(train_dataset)
    return model, history, extras


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits_list.append(outputs.cpu().numpy())
            labels_list.append(targets.cpu().numpy())
    if logits_list:
        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
    else:
        logits = np.empty((0, 2), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
    return logits, labels


def collect_logits_no_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_list: List[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits_list.append(outputs.cpu().numpy())
    if logits_list:
        return np.concatenate(logits_list, axis=0)
    return np.empty((0, 2), dtype=np.float32)


def mine_hard_negative_entries(
    model: nn.Module,
    ood_paths: Sequence[Path],
    transform: transforms.Compose,
    device: torch.device,
    top_k: int,
    threshold: float,
    exclude_paths: Set[str],
    batch_size: int,
    num_workers: int,
) -> Tuple[List[Entry], List[Dict[str, object]]]:
    if top_k <= 0 or not ood_paths:
        return [], []
    dataset = ImageOnlyDataset(ood_paths, transform, return_paths=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    candidates: List[Tuple[float, Path]] = []
    model.eval()
    with torch.no_grad():
        for images, path_batch in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            for prob_tensor, path in zip(probs, path_batch):
                score = float(prob_tensor.item())
                path_str = str(path)
                if path_str in exclude_paths or score < threshold:
                    continue
                candidates.append((score, Path(path)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_entries: List[Entry] = []
    details: List[Dict[str, object]] = []
    for score, path in candidates[:top_k]:
        selected_entries.append(Entry(path=path, target=0))
        details.append({"path": str(path), "score": float(score)})
    return selected_entries, details


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if features.size(0) < 2:
        return features.new_tensor(0.0)
    device = features.device
    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / max(temperature, 1e-6)
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logits_max.detach()
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mask_sum = mask.sum(dim=1)
    valid = mask_sum > 0
    if not torch.any(valid):
        return features.new_tensor(0.0)
    mean_log_prob_pos = torch.zeros_like(mask_sum)
    mean_log_prob_pos[valid] = (mask * log_prob).sum(dim=1)[valid] / mask_sum[valid]
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def run_supcon_stage(
    model: nn.Module,
    dataset: SonarDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if args.supcon_epochs <= 0:
        return {}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.to(device)
    projection = ProjectionHead(model.fc.out_features, args.supcon_proj_dim).to(device)
    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(projection.parameters()),
        lr=args.supcon_lr,
        weight_decay=args.weight_decay,
    )
    history: List[float] = []
    for epoch in range(1, args.supcon_epochs + 1):
        model.train()
        projection.train()
        running_loss = 0.0
        steps = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            embeddings = projection(logits)
            loss = supervised_contrastive_loss(embeddings, targets, args.supcon_temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
        epoch_loss = running_loss / max(1, steps)
        history.append(epoch_loss)
        print(f"SupCon Epoch {epoch}: loss={epoch_loss:.4f}")
    for name, param in model.named_parameters():
        param.requires_grad = original_requires_grad.get(name, True)
    projection.cpu()
    return {
        "loss": history,
        "temperature": args.supcon_temperature,
        "proj_dim": args.supcon_proj_dim,
        "epochs": args.supcon_epochs,
    }


def forward_penultimate_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Forward pass until the penultimate layer (before the final fully connected layer)."""
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def collect_penultimate_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            feats = forward_penultimate_features(model, images)
            features.append(feats.cpu().numpy())
            labels.append(targets.cpu().numpy())
    if features:
        feat_array = np.concatenate(features, axis=0)
        label_array = np.concatenate(labels, axis=0)
    else:
        feat_array = np.empty((0, model.fc.in_features if hasattr(model.fc, "in_features") else 0), dtype=np.float32)
        label_array = np.empty((0,), dtype=np.int64)
    return feat_array, label_array


def collect_penultimate_features_no_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    features: List[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            feats = forward_penultimate_features(model, images)
            features.append(feats.cpu().numpy())
    if features:
        return np.concatenate(features, axis=0)
    return np.empty((0, model.fc.in_features if hasattr(model.fc, "in_features") else 0), dtype=np.float32)


def _flatten_parameter_gradients(parameters: Sequence[torch.nn.Parameter]) -> np.ndarray:
    grads: List[torch.Tensor] = []
    for param in parameters:
        grad = param.grad
        if grad is None:
            grads.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
        else:
            grads.append(grad.detach().cpu().reshape(-1))
    if grads:
        flat = torch.cat(grads, dim=0)
        return flat.numpy()
    return np.empty((0,), dtype=np.float32)


def collect_gradient_vectors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    parameters: Optional[Sequence[torch.nn.Parameter]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    if parameters is None:
        params: List[torch.nn.Parameter] = []
        if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
            params.extend(p for p in model.fc.parameters() if p.requires_grad)
        else:
            params.extend(p for p in model.parameters() if p.requires_grad)
    else:
        params = [p for p in parameters if p.requires_grad]

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    uniform_target: Optional[torch.Tensor] = None

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
            target_tensor = targets
        else:
            images = batch
            target_tensor = None
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        for idx in range(batch_size):
            sample = images[idx : idx + 1]
            model.zero_grad(set_to_none=True)
            outputs = model(sample)
            log_probs = F.log_softmax(outputs, dim=1)
            if uniform_target is None or uniform_target.size(1) != log_probs.size(1):
                uniform_target = torch.full_like(log_probs, 1.0 / log_probs.size(1))
            loss = F.kl_div(log_probs, uniform_target, reduction="batchmean")
            loss.backward()
            vectors.append(_flatten_parameter_gradients(params))
            if target_tensor is not None:
                labels.append(int(target_tensor[idx].item()))
    model.zero_grad(set_to_none=True)
    if vectors:
        grad_array = np.stack(vectors, axis=0)
    else:
        grad_array = np.empty((0, sum(p.numel() for p in params)), dtype=np.float32)
    label_array: Optional[np.ndarray]
    if labels:
        label_array = np.asarray(labels, dtype=np.int64)
    else:
        label_array = None
    return grad_array, label_array


def fit_gaussian_mixture(
    features: np.ndarray,
    n_components: int,
    reg_covar: float,
    seed: int,
) -> Optional[GaussianMixture]:
    if features.size == 0:
        return None
    components = max(1, min(int(n_components), features.shape[0]))
    gmm = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=seed,
    )
    gmm.fit(features.astype(np.float64))
    return gmm


def gmm_max_component_log_prob(gmm: GaussianMixture, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.empty((0,), dtype=np.float64)
    data = features.astype(np.float64)
    if hasattr(gmm, "_estimate_weighted_log_prob"):
        weighted_log_prob = gmm._estimate_weighted_log_prob(data)  # type: ignore[attr-defined]
        return weighted_log_prob.max(axis=1)
    if hasattr(gmm, "_estimate_log_prob") and hasattr(gmm, "_estimate_log_weights"):
        log_prob = gmm._estimate_log_prob(data)  # type: ignore[attr-defined]
        log_weights = gmm._estimate_log_weights()  # type: ignore[attr-defined]
        return (log_prob + log_weights).max(axis=1)
    # Fallback: use mixture log probability (log-sum-exp) if component-wise API is unavailable.
    mixture_log_prob = gmm.score_samples(data)
    return mixture_log_prob


def compute_scores(logits: np.ndarray, method: str) -> np.ndarray:
    tensor = torch.from_numpy(logits)
    if method == "msp":
        probs = F.softmax(tensor, dim=1)
        scores = 1.0 - torch.max(probs, dim=1).values
    elif method == "energy":
        scores = -torch.logsumexp(tensor, dim=1)
    elif method == "entropy":
        probs = F.softmax(tensor, dim=1)
        scores = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
    elif method == "max_logit":
        scores = -torch.max(tensor, dim=1).values
    else:
        raise ValueError(f"Unknown method: {method}")
    return scores.numpy()


def fpr_at_95_tpr(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    labels = np.concatenate([np.zeros_like(in_scores), np.ones_like(ood_scores)])
    scores = np.concatenate([in_scores, ood_scores])
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    tpr_target = 0.95

    total_pos = np.sum(sorted_labels == 0)
    total_neg = np.sum(sorted_labels == 1)
    if total_pos == 0 or total_neg == 0:
        return float("nan")

    tp = total_pos
    fp = total_neg
    fpr_best = 1.0

    for thresh_idx in range(len(sorted_scores)):
        score_threshold = sorted_scores[thresh_idx]
        preds = scores >= score_threshold
        tp = np.sum((preds == 0) & (labels == 0))
        tpr = tp / total_pos
        if tpr >= tpr_target:
            fp = np.sum((preds == 0) & (labels == 1))
            fpr = fp / total_neg
            fpr_best = min(fpr_best, fpr)
    return fpr_best


def compute_ood_metrics(in_scores: np.ndarray, ood_scores: np.ndarray, method: str) -> Dict[str, float]:
    labels = np.concatenate([np.zeros_like(in_scores), np.ones_like(ood_scores)])
    scores = np.concatenate([in_scores, ood_scores])
    plt.title(f"ID scores ({method})")
    plt.hist(in_scores, alpha=0.5, label="in-distribution")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title(f"OOD scores ({method})")
    plt.hist(ood_scores, alpha=0.5, label="out-of-distribution")
    plt.legend()
    plt.grid()
    plt.show()
    
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")
    try:
        aupr = average_precision_score(labels, scores)
    except ValueError:
        aupr = float("nan")
    fpr95 = fpr_at_95_tpr(in_scores, ood_scores)
    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr_at_95_tpr": float(fpr95),
    }


def evaluate_ood_methods(
    in_logits: np.ndarray,
    ood_logits: np.ndarray,
    extra_methods: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for method in ["msp", "energy", "entropy", "max_logit"]:
        in_scores = compute_scores(in_logits, method)
        ood_scores = compute_scores(ood_logits, method)
        results[method] = compute_ood_metrics(in_scores, ood_scores, method)
    if extra_methods:
        for method, (in_scores, ood_scores) in extra_methods.items():
            results[method] = compute_ood_metrics(in_scores, ood_scores, method)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary sonar classifier with optional hard negatives, outlier exposure, SupCon fine-tuning, and evaluate OOD detection."
    )
    parser.add_argument("--data-dir", type=str, default="datasets/data_sonar")
    parser.add_argument("--output-dir", type=str, default="runs_ood")
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
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--outlier-exposure-weight", type=float, default=0.0,
                        help="Weight for KL regularisation on OOD samples (0 disables outlier exposure)")
    parser.add_argument("--hard-negative-topk", type=int, default=0,
                        help="Top-k hard negatives (object_c) to mine per epoch (0 disables hard-negative mining)")
    parser.add_argument("--hard-negative-threshold", type=float, default=0.6,
                        help="Minimum positive probability required to consider an OOD sample as hard negative")
    parser.add_argument("--hard-negative-start-epoch", type=int, default=2,
                        help="Epoch from which to start mining hard negatives")
    parser.add_argument("--hard-negative-max", type=int, default=0,
                        help="Maximum number of hard negatives to add overall (0 = unlimited)")
    parser.add_argument("--supcon-epochs", type=int, default=0,
                        help="Run an additional supervised contrastive fine-tuning stage for this many epochs")
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--supcon-proj-dim", type=int, default=64)
    parser.add_argument("--supcon-lr", type=float, default=1e-3)
    parser.add_argument("--feature-gmm-components", type=int, default=3)
    parser.add_argument("--gradient-gmm-components", type=int, default=3)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    return parser.parse_args()


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

    train_dataset = SonarDataset(train_entries, train_transform)
    val_dataset = SonarDataset(val_entries, eval_transform) if val_entries else None
    test_dataset = SonarDataset(test_entries, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ood_paths = collect_ood_samples(data_dir)
    ood_dataset = ImageOnlyDataset(ood_paths, eval_transform)
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if val_dataset is not None and len(val_dataset) > 0:
        feature_fit_dataset: Dataset = val_dataset
        feature_fit_source = "val"
    else:
        feature_fit_dataset = SonarDataset(train_entries, eval_transform)
        feature_fit_source = "train"
    feature_fit_loader = DataLoader(
        feature_fit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.freeze_backbone)
    model, history, train_extras = train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        args,
        ood_paths,
        eval_transform,
    )
    supcon_logs = run_supcon_stage(model, train_dataset, device, args)
    if supcon_logs:
        train_extras["supcon"] = supcon_logs

    test_logits, test_labels = collect_logits(model, test_loader, device)
    preds = np.argmax(test_logits, axis=1)
    acc = accuracy_score(test_labels, preds)
    report = classification_report(test_labels, preds, target_names=["background", "object_of_interest"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_labels, preds, labels=[0, 1]).tolist()

    ood_logits = collect_logits_no_labels(model, ood_loader, device)

    # ---- NEW: get features for val/train (fit), test, and ood ----
    feature_fit_features, feature_fit_labels = collect_penultimate_features(model, feature_fit_loader, device)
    test_features, _ = collect_penultimate_features(model, test_loader, device)
    ood_features = collect_penultimate_features_no_labels(model, ood_loader, device)

    ooi_proto = fit_ooi_prototype(feature_fit_features, feature_fit_labels)

    sim_val = cosine_similarity_to(ooi_proto, feature_fit_features[feature_fit_labels == 1])
    sim_test = cosine_similarity_to(ooi_proto, test_features)
    sim_ood  = cosine_similarity_to(ooi_proto, ood_features)

    tau_sim = calibrate_tau_sim(sim_val, keep_tpr=0.95)

    p_test = torch.softmax(torch.from_numpy(test_logits), dim=1)[:, 1].numpy()
    p_ood  = torch.softmax(torch.from_numpy(ood_logits),  dim=1)[:, 1].numpy()

    tau_p = 0.5  # start here; you can grid-search later
    yhat_test_gate = (p_test > tau_p) & (sim_test > tau_sim)

    # Evaluate the impact on ID test set
    gated_acc = accuracy_score(test_labels, yhat_test_gate.astype(int))
    gated_report = classification_report(
        test_labels, yhat_test_gate.astype(int),
        target_names=["background", "object_of_interest"],
        output_dict=True, zero_division=0
    )

    # ---- Treat OOD as negatives and see how many false fires remain ----
    # FP rate on OOD (lower is better)
    fp_rate_ood_gate = float(np.mean((p_ood > tau_p) & (sim_ood > tau_sim)))

    # You can also create an OOD score out of the gate (for your evaluate_ood_methods):
    # use negative cosine as a score (higher = more OOD)
    cosine_in_scores  = -sim_test
    cosine_ood_scores = -sim_ood

    extra_methods: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    extra_methods["cosine_to_ooi"] = (cosine_in_scores, cosine_ood_scores)

    ood_results = evaluate_ood_methods(test_logits, ood_logits, extra_methods=extra_methods)

    gate_metrics = {
        "tau_p": tau_p,
        "tau_sim": float(tau_sim),
        "gated_test_accuracy": float(gated_acc),
        "gated_test_report": gated_report,
        "ood_fp_rate_two_gate": fp_rate_ood_gate,
    }
    feature_gmm = fit_gaussian_mixture(
        feature_fit_features,
        args.feature_gmm_components,
        args.gmm_reg_covar,
        args.seed,
    )
    if feature_gmm is not None:
        print(
            f"Feature-space GMM fitted with {feature_gmm.n_components} components "
            f"on {feature_fit_features.shape[0]} samples ({feature_fit_source} set)."
        )
        feature_in_scores = -gmm_max_component_log_prob(feature_gmm, test_features)
        feature_ood_scores = -gmm_max_component_log_prob(feature_gmm, ood_features)
        extra_methods["feature_gmm_max_loglik"] = (feature_in_scores, feature_ood_scores)
    else:
        print("Feature-space GMM skipped: no samples available for fitting.")

    gradient_parameters = (
        list(model.fc.parameters()) if hasattr(model, "fc") and isinstance(model.fc, nn.Module) else list(model.parameters())
    )
    gradient_fit_vectors, _ = collect_gradient_vectors(model, feature_fit_loader, device, gradient_parameters)
    gradient_gmm = fit_gaussian_mixture(
        gradient_fit_vectors,
        args.gradient_gmm_components,
        args.gmm_reg_covar,
        args.seed,
    )
    if gradient_gmm is not None:
        print(
            f"Gradient-space GMM fitted with {gradient_gmm.n_components} components "
            f"on {gradient_fit_vectors.shape[0]} samples ({feature_fit_source} set)."
        )
        gradient_test_vectors, _ = collect_gradient_vectors(model, test_loader, device, gradient_parameters)
        gradient_ood_vectors, _ = collect_gradient_vectors(model, ood_loader, device, gradient_parameters)
        gradient_in_scores = -gmm_max_component_log_prob(gradient_gmm, gradient_test_vectors)
        gradient_ood_scores = -gmm_max_component_log_prob(gradient_gmm, gradient_ood_vectors)
        extra_methods["gradient_gmm_max_loglik"] = (gradient_in_scores, gradient_ood_scores)
    else:
        print("Gradient-space GMM skipped: no samples available for fitting.")

    ood_results = evaluate_ood_methods(test_logits, ood_logits, extra_methods=extra_methods)

    print("\nTraining extras:")
    print(json.dumps(train_extras, indent=2))
    print("\nIn-distribution test accuracy:", acc)
    print("Classification report:")
    print(json.dumps(report, indent=2))
    print("\nOOD results:")
    print(json.dumps(ood_results, indent=2))
    print("\nGate metrics:")
    print(json.dumps(gate_metrics, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "binary_classifier.pt")
    summary = {
        "history": history,
        "training": train_extras,
        "classification": {
            "accuracy": float(acc),
            "report": report,
            "confusion_matrix": matrix,
        },
        "ood": ood_results,
        "gate_metrics": gate_metrics,
        "ood_metadata": {
            "feature_gmm_components": int(feature_gmm.n_components) if feature_gmm is not None else 0,
            "feature_gmm_fit_samples": int(feature_fit_features.shape[0]),
            "feature_gmm_fit_source": feature_fit_source,
            "gradient_gmm_components": int(gradient_gmm.n_components) if gradient_gmm is not None else 0,
            "gradient_gmm_fit_samples": int(gradient_fit_vectors.shape[0]),
            "gradient_vector_dimension": int(gradient_fit_vectors.shape[1]) if gradient_fit_vectors.size else 0,
        },
    }
    metadata = summary.setdefault("metadata", {})
    metadata.update({
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
        "device": str(device),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "outlier_exposure_weight": args.outlier_exposure_weight,
        "label_smoothing": args.label_smoothing,
        "hard_negative_topk": args.hard_negative_topk,
        "hard_negative_threshold": args.hard_negative_threshold,
        "hard_negative_start_epoch": args.hard_negative_start_epoch,
        "hard_negative_max": args.hard_negative_max,
        "supcon_epochs": args.supcon_epochs,
        "supcon_temperature": args.supcon_temperature,
        "supcon_proj_dim": args.supcon_proj_dim,
        "weights_file": "binary_classifier.pt",
    })
    metadata["train_dataset_final_size"] = train_extras.get("final_dataset_size")
    with (output_dir / "ood_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

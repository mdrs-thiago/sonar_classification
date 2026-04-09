import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import ood_methods_extended_v2 as ood_ext

@dataclass(frozen=True)
class Sample:
    path: Path
    original_label: str
    target: int
    fine_grained_target: int

@dataclass(frozen=True)
class Entry:
    path: Path
    original_label: str
    target: int
    fine_grained_target: int

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
        return image, entry.target, entry.original_label

class OODLoaderWrapper:
    """Wraps our 3-element dataloader into a 2-element one for OOD methods."""
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for img, tgt, orig in self.loader:
            yield img, tgt

class SiamesePairDataset(Dataset):
    def __init__(self, entries: Sequence[Entry], transform: transforms.Compose, positive_ratio: float = 0.5):
        self.entries = list(entries)
        self.transform = transform
        self.positive_ratio = positive_ratio
        
        self.groups = {}
        for entry in self.entries:
            fg = entry.fine_grained_target
            if fg not in self.groups:
                self.groups[fg] = []
            self.groups[fg].append(entry)
        self.labels = list(self.groups.keys())
        
    def __len__(self) -> int:
        return len(self.entries)

    def _sample_positive(self):
        label = random.choice(self.labels)
        first, second = random.sample(self.groups[label], 2)
        return first, second, 1

    def _sample_negative(self):
        entry_a = random.choice(self.groups[0])
        entry_b = random.choice(self.groups[1])
        return entry_a, entry_b, 0

    def __getitem__(self, idx: int):
        if random.random() < self.positive_ratio:
            entry_a, entry_b, target = self._sample_positive()
        else:
            entry_a, entry_b, target = self._sample_negative()
            
        img_a = Image.open(entry_a.path).convert("RGB")
        img_b = Image.open(entry_b.path).convert("RGB")
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, torch.tensor(target, dtype=torch.float32)

class EmbeddingNet(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int, freeze_backbone: bool = False):
        super().__init__()
        if model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.fc.in_features
        elif model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.fc.in_features
        elif model_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            self.backbone = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1))
            in_features = backbone.classifier[0].in_features
        else:
            raise ValueError(f"Unknown Siamese model: {model_name}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        distances = F.pairwise_distance(output1, output2)
        positive_loss = target * (distances ** 2)
        negative_loss = (1 - target) * torch.clamp(self.margin - distances, min=0.0) ** 2
        return torch.mean(positive_loss + negative_loss)

def build_classifier(model_name: str, freeze_backbone: bool = False) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Unknown classifier model: {model_name}")
        
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not ("fc" in name or "classifier" in name):
                param.requires_grad = False
    return model

def get_target_layer_for_gradcam(model, model_name: str):
    if "resnet" in model_name:
        return model.layer4
    elif "mobilenet" in model_name:
        return model.features[-1]
    return list(model.children())[-2]

class SpatialAttentionLoss:
    def __init__(self, target_layer):
        self.activations = None
        self.handle = target_layer.register_forward_hook(self.hook)
        
    def hook(self, module, input, output):
        self.activations = output
        
    def compute_loss(self, orig_labels, device):
        if self.activations is None:
            return torch.tensor(0.0, device=device)
        B, C, H, W = self.activations.shape
        half_w = W // 2
        loss = torch.tensor(0.0, device=device)
        count = 0
        
        for i, label in enumerate(orig_labels):
            if label == "object_a": # Left side
                penalty = self.activations[i, :, :, half_w:].pow(2).mean()
                loss += penalty
                count += 1
            elif label == "object_b": # Right side
                penalty = self.activations[i, :, :, :half_w].pow(2).mean()
                loss += penalty
                count += 1
        return loss / max(1, count)
        
    def remove(self):
        self.handle.remove()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_samples(root: Path) -> Tuple[List[Sample], List[Sample]]:
    mapping = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 1,
    }
    fine_grained_mapping = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 2,
    }
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    in_dist_samples = []
    ood_samples = []
    
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        original_label = class_dir.name
        if original_label == "object_c":
            for fp in sorted(class_dir.iterdir()):
                if fp.suffix.lower() in exts:
                    ood_samples.append(Sample(path=fp, original_label=original_label, target=-1, fine_grained_target=-1))
            continue
        
        if original_label not in mapping:
            continue
            
        target = mapping[original_label]
        fg_target = fine_grained_mapping[original_label]
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() in exts:
                in_dist_samples.append(Sample(path=fp, original_label=original_label, target=target, fine_grained_target=fg_target))
                
    return in_dist_samples, ood_samples

def split_indices(samples: List[Sample], test_size: float = 0.2, seed: int = 42):
    indices = np.arange(len(samples))
    labels = np.array([s.target for s in samples])
    
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=labels, random_state=seed)
    return train_idx.tolist(), test_idx.tolist()

def make_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform

def train_classifier(model, train_loader, device, epochs=10, lr=3e-4, use_spatial_loss=True, model_name="resnet18"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    target_layer = get_target_layer_for_gradcam(model, model_name)
    att_loss_fn = SpatialAttentionLoss(target_layer) if use_spatial_loss else None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, orig_labels in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            if use_spatial_loss:
                # Add guided attention penalty
                sp_loss = att_loss_fn.compute_loss(orig_labels, device)
                loss += 0.5 * sp_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Classifier Epoch {epoch}/{epochs} Loss: {running_loss/len(train_loader):.4f}")
        
    if use_spatial_loss:
        att_loss_fn.remove()
    return model

def train_siamese(model, train_loader, device, epochs=10, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=1.0)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for img_a, img_b, targets in train_loader:
            img_a, img_b, targets = img_a.to(device), img_b.to(device), targets.to(device)
            optimizer.zero_grad()
            emb_a, emb_b = model(img_a), model(img_b)
            loss = criterion(emb_a, emb_b, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Siamese Epoch {epoch}/{epochs} Loss: {running_loss/len(train_loader):.4f}")
    return model

def extract_siamese_prototypes(model, loader, device, num_classes=3):
    model.eval()
    embeddings_by_class = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for images, _, orig_labels in loader:
            images = images.to(device)
            embs = model(images)
            for emb, orig in zip(embs, orig_labels):
                fg = 1 if orig == "object_a" else 2 if orig == "object_b" else 0
                embeddings_by_class[fg].append(emb.cpu())
                
    prototypes = {}
    for cls in range(num_classes):
        if len(embeddings_by_class[cls]) > 0:
            stacked = torch.stack(embeddings_by_class[cls])
            prototypes[cls] = F.normalize(stacked.mean(dim=0), p=2, dim=0).to(device)
    return prototypes

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, input_tensor, target_class=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, target_class].sum()
        score.backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

def compute_center_of_mass(heatmap: np.ndarray):
    heatmap = np.clip(heatmap, 0, 1)
    total_mass = heatmap.sum()
    if total_mass == 0:
        return 0, 0
    y_indices, x_indices = np.indices(heatmap.shape)
    x_center = (heatmap * x_indices).sum() / total_mass
    y_center = (heatmap * y_indices).sum() / total_mass
    return x_center, y_center

from sklearn.metrics import roc_curve, average_precision_score

def compute_ood_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx[0]] if len(idx) > 0 else 1.0
    return {"AUROC": auroc, "AUPR": aupr, "FPR95": fpr95}

def evaluate_extended_ood(classifier, siamese, prototypes, in_dist_loader, ood_loader, device):
    classifier.eval()
    siamese.eval()
    
    in_dist_wrap = OODLoaderWrapper(in_dist_loader)
    ood_wrap = OODLoaderWrapper(ood_loader)
    
    results = {}
    
    # 1. Siamese metrics (manual evaluation based on distances)
    def compute_siamese_scores(loader, is_in_dist):
        scores_sim, scores_euclid = [], []
        labels = []
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(device)
                embs = siamese(images)
                
                # Max cosine (higher => closer => ID)
                sims = [F.cosine_similarity(embs, proto.unsqueeze(0)) for proto in prototypes.values()]
                max_sim = torch.max(torch.stack(sims), dim=0).values.cpu().numpy()
                scores_sim.extend(max_sim)
                
                # Negative min euclidean (higher => closer => ID)
                dists = [F.pairwise_distance(embs, proto.unsqueeze(0)) for proto in prototypes.values()]
                min_dist = torch.min(torch.stack(dists), dim=0).values.cpu().numpy()
                scores_euclid.extend(-min_dist)
                
                labels.extend([1 if is_in_dist else 0] * len(images))
        return scores_sim, scores_euclid, labels

    sim_in, euc_in, lab_in = compute_siamese_scores(in_dist_loader, True)
    sim_out, euc_out, lab_out = compute_siamese_scores(ood_loader, False)
    
    all_sim = np.array(sim_in + sim_out) + np.random.uniform(0, 1e-7, size=len(sim_in)+len(sim_out))
    all_euc = np.array(euc_in + euc_out) + np.random.uniform(0, 1e-7, size=len(euc_in)+len(euc_out))
    all_labels = lab_in + lab_out
    
    results["SiameseCosine"] = compute_ood_metrics(all_labels, all_sim)
    results["SiameseEuclidean"] = compute_ood_metrics(all_labels, all_euc)
    
    # 2. Extract and evaluate extended classifier OOD methods
    methods = {
        "MSP": ood_ext.MSP(classifier),
        "Energy": ood_ext.EnergyBased(classifier),
        "FeatureMahalanobis": ood_ext.FeatureMahalanobis(classifier),
        "GradNorm": ood_ext.GradNorm(classifier),
        "TwoSidedHeadGradResidual": ood_ext.TwoSidedHeadGradResidual(classifier)
    }

    print("Fitting OOD methods...")
    for name, method in methods.items():
        if hasattr(method, "fit"):
            try:
                method.fit(in_dist_wrap)
            except Exception as e:
                print(f"Failed to fit {name}: {e}")
                
    for name, method in methods.items():
        print(f"Scoring {name}...")
        try:
            scores_in = method.compute_ood_scores(in_dist_wrap).numpy()
            scores_out = method.compute_ood_scores(ood_wrap).numpy()
            
            # ALL methods in ood_methods_extended_v2 return HIGHER = OOD.
            # To align with our labels (ID=1, OOD=0), we want HIGHER = ID.
            # So we invert all scores returned by these methods.
            scores_in = -scores_in
            scores_out = -scores_out
                
            labels = [1]*len(scores_in) + [0]*len(scores_out)
            all_scores = np.concatenate([scores_in, scores_out]) + np.random.uniform(0, 1e-7, size=len(scores_in)+len(scores_out))
            results[name] = compute_ood_metrics(labels, all_scores)
        except Exception as e:
            print(f"Skipping {name} due to score extraction error: {e}")
            
    return results

def evaluate_localization(classifier, dataset: SonarDataset, device, model_name: str):
    classifier.eval()
    layer = get_target_layer_for_gradcam(classifier, model_name)
    grad_cam = GradCAM(classifier, layer)
    
    preds_left_right = []
    true_left_right = []
    
    for entry in dataset.entries:
        if entry.target != 1:  # Only look at objects!
            continue
            
        is_object_a = (entry.original_label == "object_a")
        
        image = Image.open(entry.path).convert("RGB")
        resize_tf = transforms.Resize((224, 224))
        image = resize_tf(image)
        
        tensor = transforms.ToTensor()(image)
        tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
        tensor = tensor.unsqueeze(0).to(device)
        
        heatmap = grad_cam.generate(tensor, target_class=1)
        x_c, y_c = compute_center_of_mass(heatmap)
        
        # Determine if left or right based on center x.
        predicted_is_a = (x_c < 112)
        
        preds_left_right.append(predicted_is_a)
        true_left_right.append(is_object_a)
        
    grad_cam.remove()
    
    acc = accuracy_score(true_left_right, preds_left_right)
    final_acc = max(acc, 1 - acc)
    return final_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v3_small"])
    parser.add_argument("--no-spatial-loss", action="store_true", help="Disable spatial attention loss")
    args = parser.parse_args()
    
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Model: {args.model_name}")
    
    in_dist_samples, ood_samples = collect_samples(Path(args.data_dir))
    print(f"In-dist samples: {len(in_dist_samples)}, OOD samples: {len(ood_samples)}")
    if len(in_dist_samples) == 0:
        raise ValueError("No in-distribution samples found. Check --data-dir path.")
    
    train_idx, test_idx = split_indices(in_dist_samples, seed=seed)
    
    train_entries = [Entry(path=in_dist_samples[i].path, original_label=in_dist_samples[i].original_label, target=in_dist_samples[i].target, fine_grained_target=in_dist_samples[i].fine_grained_target) for i in train_idx]
    test_entries = [Entry(path=in_dist_samples[i].path, original_label=in_dist_samples[i].original_label, target=in_dist_samples[i].target, fine_grained_target=in_dist_samples[i].fine_grained_target) for i in test_idx]
    ood_entries = [Entry(path=s.path, original_label=s.original_label, target=s.target, fine_grained_target=s.fine_grained_target) for s in ood_samples]
    
    train_tf, eval_tf = make_transforms()
    
    classifier_train_ds = SonarDataset(train_entries, train_tf)
    classifier_test_ds = SonarDataset(test_entries, eval_tf)
    ood_ds = SonarDataset(ood_entries, eval_tf)
    
    train_loader = DataLoader(classifier_train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(classifier_test_ds, batch_size=32, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False, num_workers=0)
    
    classifier = build_classifier(args.model_name)
    print("Training Binary Classifier with Spatial Guided Attention Loss...")
    classifier = train_classifier(classifier, train_loader, device, epochs=args.epochs, use_spatial_loss=not args.no_spatial_loss, model_name=args.model_name)
    
    siamese_train_ds = SiamesePairDataset(train_entries, train_tf)
    siamese_train_loader = DataLoader(siamese_train_ds, batch_size=32, shuffle=True, num_workers=0)
    
    siamese = EmbeddingNet(model_name=args.model_name, embedding_dim=128)
    print("Training Siamese Network...")
    siamese = train_siamese(siamese, siamese_train_loader, device, epochs=args.epochs)
    print("Extracting Siamese Prototypes...")
    proto_loader = DataLoader(SonarDataset(train_entries, eval_tf), batch_size=32, shuffle=False)
    prototypes = extract_siamese_prototypes(siamese, proto_loader, device)
    
    print("Evaluating OOD Robustness...")
    ood_results = evaluate_extended_ood(classifier, siamese, prototypes, test_loader, ood_loader, device)
    for method_name, metrics in ood_results.items():
        print(f"OOD Results - {method_name}: AUROC: {metrics['AUROC']:.4f}, AUPR: {metrics['AUPR']:.4f}, FPR95: {metrics['FPR95']:.4f}")
    
    print("Evaluating Weakly Supervised Localization (Grad-CAM)...")
    loc_acc = evaluate_localization(classifier, classifier_test_ds, device, args.model_name)
    print(f"Left/Right Localization Accuracy: {loc_acc:.4f}")
    
    results = {
        "model_name": args.model_name,
        "spatial_loss_enabled": not args.no_spatial_loss,
        "ood_metrics": ood_results,
        "localization_accuracy": loc_acc
    }
    
    output_filename = f"experiment_results_{args.model_name}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Experiment finished. Results saved to {output_filename}")

if __name__ == "__main__":
    main()

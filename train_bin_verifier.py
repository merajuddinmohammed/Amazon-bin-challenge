"""
Bin Order Verifier Training Script with Anti-Overfitting Upgrades

Dataset layout (on Windows):

C:/Users/meraj/OneDrive/Desktop/Amazon bin assignment/dataset/
    bin-images/
        00001.jpg
        ...
    metadata/
        00001.json
        ...

Each JSON contains:
{
  "BIN_FCSKU_DATA": {
      "<ASIN>": {"quantity": int, ...},
      ...
  },
  "EXPECTED_QUANTITY": int
}

We create training samples of the form:
(image, asin_id, requested_qty) -> label (1 = correct, 0 = wrong)
and train a binary classifier.
"""

import os
import json
import random
import argparse
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------- Reproducibility ----------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- Dataset Utilities ----------------- #

def load_all_metadata(meta_dir: Path):
    """
    Scan all JSON files and build:
      - meta_dict: {basename: json_dict}
      - asin2id:   {asin: idx}
      - max_qty:   max quantity value over all items
    """
    meta_paths = sorted(meta_dir.glob("*.json"))
    if len(meta_paths) == 0:
        raise RuntimeError(f"No JSON files found in {meta_dir}")

    asin_set = set()
    max_qty = 0
    meta_dict = {}

    total = len(meta_paths)
    print_every = max(1, total // 10)

    for idx, mp in enumerate(meta_paths):
        with open(mp, "r") as f:
            data = json.load(f)

        basename = mp.stem
        meta_dict[basename] = data

        bin_data = data.get("BIN_FCSKU_DATA", {})
        for asin, item_data in bin_data.items():
            asin_set.add(asin)
            qty = int(item_data.get("quantity", 0))
            max_qty = max(max_qty, qty)

        if (idx + 1) % print_every == 0 or (idx + 1) == total:
            print(f"  Loaded {idx+1}/{total} metadata files...", flush=True)

    asin_list = sorted(list(asin_set))
    asin2id = {asin: idx for idx, asin in enumerate(asin_list)}

    if max_qty <= 0:
        max_qty = 10

    return meta_dict, asin2id, max_qty


def split_basenames(basenames, train_ratio=0.7, val_ratio=0.15):
    set_seed(42)
    basenames = list(basenames)
    random.shuffle(basenames)
    n_total = len(basenames)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_b = basenames[:n_train]
    val_b = basenames[n_train:n_train + n_val]
    test_b = basenames[n_train + n_val:]

    return train_b, val_b, test_b


def build_samples_for_split(
    split_basenames,
    meta_dict,
    asin2id,
    all_asins,
    max_qty,
    neg_per_bin_asin=1,
    neg_fake_asin=2,
):
    """
    Create sample dicts for a batch:

    Positive:
      (basename, asin_true, true_qty, label=1)

    Negative (wrong quantity, same ASIN):
      (basename, asin_true, wrong_qty, label=0)

    Negative (ASIN not in bin):
      (basename, asin_fake, any_qty, label=0)
    """
    samples = []

    total_bins = len(split_basenames)
    report_every = max(1, total_bins // 10)

    for i, basename in enumerate(split_basenames, start=1):
        data = meta_dict[basename]
        bin_data = data.get("BIN_FCSKU_DATA", {})

        if not bin_data:
            continue

        # Positive + negative-qty for ASINs that exist in this bin
        for asin_true, item_data in bin_data.items():
            if asin_true not in asin2id:
                continue

            asin_id = asin2id[asin_true]
            qty_true = int(item_data.get("quantity", 0))
            if qty_true <= 0:
                continue

            # Positive sample
            samples.append(
                {
                    "basename": basename,
                    "asin_id": asin_id,
                    "qty": qty_true,
                    "label": 1,
                }
            )

            # Negative quantity samples
            for _ in range(neg_per_bin_asin):
                q_neg = qty_true
                tries = 0
                while q_neg == qty_true and tries < 10:
                    q_neg = random.randint(1, max_qty)
                    tries += 1
                if q_neg != qty_true:
                    samples.append(
                        {
                            "basename": basename,
                            "asin_id": asin_id,
                            "qty": q_neg,
                            "label": 0,
                        }
                    )

        # Negative ASIN samples: ASIN not present in this bin
        bin_asins = set(bin_data.keys())
        other_asins = [a for a in all_asins if a not in bin_asins]
        if len(other_asins) == 0:
            continue

        for _ in range(neg_fake_asin):
            asin_fake = random.choice(other_asins)
            asin_id_fake = asin2id[asin_fake]
            q_fake = random.randint(1, max_qty)
            samples.append(
                {
                    "basename": basename,
                    "asin_id": asin_id_fake,
                    "qty": q_fake,
                    "label": 0,
                }
            )

        if i % report_every == 0 or i == total_bins:
            print(
                f"  Built samples for {i}/{total_bins} bins ({len(samples)} samples total)...",
                flush=True,
            )

    return samples


class BinOrderDataset(Dataset):
    """
    Dataset of (image, asin_id, qty_norm, label)
    """

    def __init__(self, samples, img_dir, max_qty, transform=None):
        self.samples = samples
        self.img_dir = Path(img_dir)
        self.max_qty = max_qty
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        basename = s["basename"]
        asin_id = s["asin_id"]
        qty = s["qty"]
        label = s["label"]

        img_path = self.img_dir / f"{basename}.jpg"
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Normalize quantity to [0, 1]
        qty_norm = float(qty) / float(self.max_qty)

        return (
            img,
            torch.tensor(asin_id, dtype=torch.long),
            torch.tensor([qty_norm], dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ----------------- Model ----------------- #

class BinOrderVerifier(nn.Module):
    """
    Image (ResNet18) + ASIN embedding + quantity MLP -> binary decision
    """

    def __init__(self, num_asins, asin_emb_dim=64, qty_hidden_dim=32, dropout_p=0.4):
        super().__init__()

        # Image backbone (pretrained ResNet18)
        backbone = models.resnet18(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ASIN embedding
        self.asin_emb = nn.Embedding(num_asins, asin_emb_dim)

        # Quantity branch
        self.qty_mlp = nn.Sequential(
            nn.Linear(1, qty_hidden_dim),
            nn.ReLU(),
        )

        # Classifier with higher dropout to combat overfitting
        self.classifier = nn.Sequential(
            nn.Linear(in_features + asin_emb_dim + qty_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1),
        )

    def forward(self, imgs, asin_ids, qty_norm):
        img_feat = self.backbone(imgs)                 # [B, 512]
        asin_feat = self.asin_emb(asin_ids)            # [B, asin_emb_dim]
        qty_feat = self.qty_mlp(qty_norm)              # [B, qty_hidden_dim]

        x = torch.cat([img_feat, asin_feat, qty_feat], dim=1)
        logits = self.classifier(x)                    # [B, 1]
        return logits.squeeze(1)


# ----------------- Train / Eval Loops ----------------- #

def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=5.0):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc="[Train]", leave=False)
    for imgs, asin_ids, qty_norm, labels in pbar:
        imgs = imgs.to(device)
        asin_ids = asin_ids.to(device)
        qty_norm = qty_norm.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs, asin_ids, qty_norm)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits)
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    return epoch_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc="[Eval ]", leave=False)
    with torch.no_grad():
        for imgs, asin_ids, qty_norm, labels in pbar:
            imgs = imgs.to(device)
            asin_ids = asin_ids.to(device)
            qty_norm = qty_norm.to(device)
            labels = labels.to(device)

            logits = model(imgs, asin_ids, qty_norm)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    return epoch_loss, acc, all_labels, all_probs


# ----------------- Plotting ----------------- #

def plot_training_curves(history, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "loss_curve.png", bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "accuracy_curve.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, out_dir: Path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()


def plot_roc_pr(y_true, y_probs, out_dir: Path):
    from sklearn.metrics import roc_curve, precision_recall_curve

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
        plt.close()
    except Exception:
        roc_auc = float("nan")

    # Precision–Recall
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(out_dir / "pr_curve.png", bbox_inches="tight")
        plt.close()
    except Exception:
        ap = float("nan")

    return roc_auc, ap


# ----------------- Early Stopping ----------------- #

class EarlyStopping:
    """
    Stop training when validation loss does not improve after `patience` epochs.
    """

    def __init__(self, patience=4, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ----------------- Metadata Cache Helper ----------------- #

def load_or_cache_metadata(meta_dir: Path, cache_path: Path):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"Loaded cached metadata from {cache_path}")
        return cached["meta_dict"], cached["asin2id"], cached["max_qty"]

    meta_dict, asin2id, max_qty = load_all_metadata(meta_dir)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {"meta_dict": meta_dict, "asin2id": asin2id, "max_qty": max_qty},
            f,
        )
    print(f"Saved metadata cache to {cache_path}")
    return meta_dict, asin2id, max_qty


# ----------------- Main ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="C:/Users/meraj/OneDrive/Desktop/Amazon bin assignment/dataset",
        help="Root folder containing bin-images and metadata",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="C:/Users/meraj/OneDrive/Desktop/Amazon bin assignment/results",
        help="Directory to save models and plots",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)  # 0 for Windows
    args = parser.parse_args()

    set_seed(42)

    data_root = Path(args.data_root)
    img_dir = data_root / "bin-images"
    meta_dir = data_root / "metadata"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load metadata & ASIN vocabulary (with caching)
    cache_path = data_root / ".metadata_cache.pkl"
    meta_dict, asin2id, max_qty = load_or_cache_metadata(meta_dir, cache_path)

    all_basenames = sorted(meta_dict.keys())
    all_asins = sorted(list(asin2id.keys()))
    num_asins = len(asin2id)

    print(f"Total bins: {len(all_basenames)}")
    print(f"Distinct ASINs: {num_asins}")
    print(f"Max quantity observed: {max_qty}")

    # 2) Split bins into train/val/test
    train_b, val_b, test_b = split_basenames(all_basenames)
    print(f"Train bins: {len(train_b)}, Val bins: {len(val_b)}, Test bins: {len(test_b)}")

    # 3) Build samples
    train_samples = build_samples_for_split(
        train_b, meta_dict, asin2id, all_asins, max_qty
    )
    val_samples = build_samples_for_split(
        val_b, meta_dict, asin2id, all_asins, max_qty
    )
    test_samples = build_samples_for_split(
        test_b, meta_dict, asin2id, all_asins, max_qty
    )

    print(
        f"Train samples: {len(train_samples)}, "
        f"Val samples: {len(val_samples)}, "
        f"Test samples: {len(test_samples)}"
    )

    # 4) Transforms (lighter augmentation for faster training)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # 5) Datasets & Dataloaders
    train_ds = BinOrderDataset(train_samples, img_dir, max_qty, transform=train_tf)
    val_ds = BinOrderDataset(val_samples, img_dir, max_qty, transform=eval_tf)
    test_ds = BinOrderDataset(test_samples, img_dir, max_qty, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 6) Model, loss, optimizer, scheduler, early stopping
    model = BinOrderVerifier(num_asins=num_asins, dropout_p=0.4).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,  # L2 regularization
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    early_stopper = EarlyStopping(patience=5, min_delta=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = results_dir / "best_verifier.pt"

    # 7) Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = eval_one_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}"
        )

        # Scheduler uses validation accuracy (max mode)
        scheduler.step(val_acc)

        # Save best model by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_acc = {best_val_acc:.4f}")

        # Early stopping based on validation loss (min mode)
        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break

    # 8) Training curves
    plot_training_curves(history, results_dir)

    # 9) Final evaluation on test set
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_true, y_probs = eval_one_epoch(
        model, test_loader, criterion, device
    )
    y_pred = (y_probs >= 0.5).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_probs)
    except Exception:
        ap = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, results_dir)
    roc_auc_plot, ap_plot = plot_roc_pr(y_true, y_probs, results_dir)

    # 10) Write metrics to file
    metrics_path = results_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("=== Best Validation ===\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n\n")

        f.write("=== Test Metrics ===\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {prec:.4f}\n")
        f.write(f"Test Recall: {rec:.4f}\n")
        f.write(f"Test F1-score: {f1:.4f}\n")
        f.write(f"Test ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Test Average Precision (PR AUC): {ap:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print("\n===== FINAL TEST METRICS =====")
    print(open(metrics_path).read())
    print(f"All artifacts saved in: {results_dir}")


if __name__ == "__main__":
    main()

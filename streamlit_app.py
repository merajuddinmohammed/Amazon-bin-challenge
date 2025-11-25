import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import streamlit as st

# ================== Paths ==================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATASET_DIR = BASE_DIR / "dataset"
META_CACHE_PATH = DATASET_DIR / ".metadata_cache.pkl"
MODEL_PATH = RESULTS_DIR / "best_verifier.pt"

# ================== App Layout ==================

st.set_page_config(
    page_title="Bin Order Verifier",
    layout="wide",
    page_icon="📦",
)

st.title("📦 Bin Order Verifier")
st.markdown("Verify if a bin image matches the expected **ASIN** and **quantity**.")
st.markdown("---")


# ================== Model Definition ==================

class BinOrderVerifier(nn.Module):
    def __init__(self, num_asins: int, asin_emb_dim: int = 64,
                 qty_hidden_dim: int = 32, dropout_p: float = 0.4):
        super().__init__()

        backbone = models.resnet18(weights=None)   # weights loaded via training script
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.asin_emb = nn.Embedding(num_asins, asin_emb_dim)
        self.qty_mlp = nn.Sequential(
            nn.Linear(1, qty_hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + asin_emb_dim + qty_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 1),
        )

    def forward(self, imgs, asin_ids, qty_norm):
        img_feat = self.backbone(imgs)
        asin_feat = self.asin_emb(asin_ids)
        qty_feat = self.qty_mlp(qty_norm)
        x = torch.cat([img_feat, asin_feat, qty_feat], dim=1)
        logits = self.classifier(x)
        return logits.squeeze(1)


# ================== Helpers ==================

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def safe_open_pil(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Could not open image. Please upload a valid JPG/PNG file.")
        return None
    return img


# ================== Load Model & Metadata ==================

@st.cache_resource(show_spinner=True)
def load_model_and_metadata():
    """Load trained model, ASIN mapping and quantity scaling info."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata cache
    if not META_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"metadata_cache.pkl not found at: {META_CACHE_PATH}. "
            f"Make sure the dataset folder has the .metadata_cache.pkl file."
        )

    with open(META_CACHE_PATH, "rb") as f:
        cached = pickle.load(f)

    asin2id = cached["asin2id"]
    id2asin = {v: k for k, v in asin2id.items()}
    max_qty = float(cached["max_qty"])

    # Init and load model
    model = BinOrderVerifier(num_asins=len(asin2id), dropout_p=0.4)
    best_model_path = RESULTS_DIR / "best_verifier.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"best_verifier.pt not found at: {best_model_path}. "
            f"Make sure the results folder has your trained model."
        )

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = build_transform()

    return model, device, asin2id, id2asin, max_qty, transform


def predict(model, device, transform, image, asin_id, quantity, max_qty):
    """Run model for a single example and return probability of 'CORRECT'."""
    img_tensor = transform(image).unsqueeze(0).to(device)
    asin_tensor = torch.tensor([asin_id], dtype=torch.long, device=device)
    qty_norm = float(quantity) / max_qty
    qty_tensor = torch.tensor([[qty_norm]], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(img_tensor, asin_tensor, qty_tensor)
        prob = torch.sigmoid(logits).item()

    return float(prob)


# ================== Load resources ==================

try:
    model, device, asin2id, id2asin, max_qty, transform = load_model_and_metadata()
except Exception as e:
    st.error(f"Failed to load model or metadata: {e}")
    st.stop()

tabs = st.tabs(["🔍 Verify Order", "📊 Metrics", "ℹ️ About"])


# ================== TAB 1: Verify Order ==================

with tabs[0]:
    st.subheader("🔍 Verify a Bin Image")

    left_col, right_col = st.columns([1.1, 1.2])

    with left_col:
        st.markdown("#### 1. Upload Bin Image")
        uploaded = st.file_uploader("Choose a bin image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        img = None
        if uploaded is not None:
            img = safe_open_pil(uploaded)
            if img is not None:
                st.image(img, caption="Uploaded Image", use_column_width=True)

    with right_col:
        st.markdown("#### 2. Enter Expected Item & Quantity")

        asin_list = sorted(asin2id.keys())
        # For large asin list, a text input might be better; keeping selectbox for now
        selected_asin = st.selectbox("ASIN", asin_list)
        asin_id = asin2id[selected_asin]

        qty = st.number_input(
            "Expected quantity",
            min_value=1,
            max_value=int(max_qty),
            value=1,
            step=1,
        )

        st.markdown("#### 3. Run Verification")

        can_run = img is not None

        if not can_run:
            st.info("Upload an image first to enable verification.")

        if can_run and st.button("🔍 Verify Order", type="primary"):
            with st.spinner("Running model..."):
                prob_correct = predict(
                    model=model,
                    device=device,
                    transform=transform,
                    image=img,
                    asin_id=asin_id,
                    quantity=qty,
                    max_qty=max_qty,
                )

            decision_correct = prob_correct >= 0.5
            prob_wrong = 1.0 - prob_correct

            st.success("Verification complete.")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Confidence: CORRECT", f"{prob_correct * 100:.2f}%")
            with c2:
                label = "✅ CORRECT" if decision_correct else "❌ WRONG"
                st.metric("Decision", label)

            st.markdown("##### Prediction breakdown")
            details_df = pd.DataFrame(
                {
                    "Metric": [
                        "ASIN",
                        "Expected quantity",
                        "Confidence (Correct)",
                        "Confidence (Wrong)",
                        "Final decision",
                    ],
                    "Value": [
                        selected_asin,
                        f"{qty} units",
                        f"{prob_correct * 100:.2f}%",
                        f"{prob_wrong * 100:.2f}%",
                        "✅ CORRECT" if decision_correct else "❌ WRONG",
                    ],
                }
            )
            st.dataframe(details_df, hide_index=True)


# ================== TAB 2: Metrics ==================

with tabs[1]:
    st.subheader("📊 Model Performance")

    metrics_path = RESULTS_DIR / "metrics.txt"
    if metrics_path.exists():
        st.markdown("#### Test metrics")
        with open(metrics_path, "r") as f:
            st.text(f.read())
    else:
        st.warning(f"`metrics.txt` not found at {metrics_path}")

    st.markdown("#### Training & Evaluation Plots")
    col1, col2 = st.columns(2)

    with col1:
        loss_path = RESULTS_DIR / "loss_curve.png"
        acc_path = RESULTS_DIR / "accuracy_curve.png"
        if loss_path.exists():
            st.image(str(loss_path), caption="Loss Curve")
        if acc_path.exists():
            st.image(str(acc_path), caption="Accuracy Curve")

    with col2:
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        roc_path = RESULTS_DIR / "roc_curve.png"
        pr_path = RESULTS_DIR / "pr_curve.png"

        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix")
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve")
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision–Recall Curve")


# ================== TAB 3: About ==================

with tabs[2]:
    st.subheader("ℹ️ About This Project")

    st.markdown(
        """
        ### Model Architecture

        - **Backbone:** ResNet-18 (ImageNet-pretrained during training, fine-tuned on bin images)  
        - **ASIN Branch:** 64-dimensional embedding for each product ASIN  
        - **Quantity Branch:** Small MLP over normalized quantity  
        - **Head:** MLP with dropout (0.4) for robust binary classification  

        ### Training Setup

        - Optimizer: Adam with weight decay  
        - Loss: Binary Cross-Entropy with logits  
        - Scheduler: ReduceLROnPlateau on validation accuracy  
        - Early stopping based on validation loss  

        ### Dataset Summary

        - Total bins: ~9,195  
        - Distinct ASINs: ~15,231  
        - Split: 70% train / 15% val / 15% test  
        - Samples built from positive (correct) and negative (wrong ASIN / wrong qty) examples  

        ### Typical Test Performance (15 epochs)

        - Accuracy ≈ **95.1%**  
        - Precision ≈ **91.7%**  
        - Recall ≈ **95.1%**  
        - F1-score ≈ **93.3%**  
        - ROC AUC ≈ **0.986**  
        - PR AUC ≈ **0.967**  

        ### Decision Rule

        The model outputs a probability that the order is **correct**.  
        - Probability ≥ 0.5 → **CORRECT** ✅  
        - Probability < 0.5 → **WRONG** ❌  
        """
    )

st.markdown("---")
st.caption(f"Running on: `{device}` • Model: BinOrderVerifier • Max qty: {int(max_qty)}")


# 📦 Amazon Bin Order Verifier

AI-powered system to verify if a bin image matches the expected ASIN and quantity.

## Features

✅ **Deep Learning Model** - ResNet18 backbone with ASIN embeddings  
✅ **High Accuracy** - 95.06% test accuracy, 98.56% ROC AUC  
✅ **GPU Accelerated** - CUDA support for fast inference  
✅ **Streamlit Interface** - Easy-to-use web UI  
✅ **Real-time Predictions** - Get results in seconds  

## Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 95.06% |
| Precision | 91.72% |
| Recall | 95.05% |
| F1-Score | 93.35% |
| ROC AUC | 98.56% |
| PR AUC | 96.73% |

## Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/merajuddinmohammed/Amazon-bin-challenge.git
cd Amazon-bin-challenge

# Create virtual environment
python -m venv .venv

# Activate venv (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Setup (CUDA 12.1)

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

## Usage

### Run Training

```bash
python train_bin_verifier.py --epochs 15 --batch-size 32
```

**Options:**
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)

### Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Opens at: `http://localhost:8501`

## Project Structure

```
.
├── train_bin_verifier.py      # Training script
├── streamlit_app.py            # Streamlit web interface
├── requirements.txt            # Python dependencies
├── dataset/
│   ├── bin-images/            # Bin product images
│   ├── metadata/              # ASIN & quantity annotations
│   └── .metadata_cache.pkl    # Cached metadata
└── results/
    ├── best_verifier.pt       # Trained model weights
    ├── loss_curve.png         # Training curves
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    └── metrics.txt            # Final metrics
```

## Model Architecture

```
Input: (Image, ASIN, Quantity)
  ↓
ResNet18 Backbone          → Image Features (512-dim)
ASIN Embedding             → ASIN Features (64-dim)
Quantity MLP               → Quantity Features (32-dim)
  ↓
Concatenate Features (608-dim)
  ↓
Classification Head
  Linear(608, 256) + ReLU + Dropout(0.4)
  Linear(256, 1)
  ↓
Sigmoid → Probability (0-1)
  ↓
Decision: ≥0.5 → CORRECT, <0.5 → WRONG
```

## Data Preprocessing

- **Images**: Resized to 224×224, normalized with ImageNet stats
- **ASIN**: Learned 64-dimensional embeddings
- **Quantity**: Normalized to [0, 1] range based on max observed quantity
- **Augmentation**: RandomHorizontalFlip, RandomRotation for training

## Training Details

- **Optimizer**: Adam with L2 regularization (weight_decay=1e-4)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: patience=5 epochs
- **Device**: GPU (CUDA) if available, CPU fallback

## Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Select repo and deploy

### Docker

```bash
docker build -t bin-verifier .
docker run -p 8501:8501 bin-verifier
```

### Hugging Face Spaces

Push to GitHub, then create a Space on [huggingface.co/spaces](https://huggingface.co/spaces)

## Results

**Best Validation Accuracy: 94.62%**  
**Final Test Accuracy: 95.06%**

Training converged smoothly over 15 epochs without significant overfitting.

## Requirements

- Python 3.10+
- PyTorch 2.5.1 (GPU or CPU)
- Streamlit 1.51.0
- CUDA 12.1+ (for GPU acceleration)

## License

MIT License

## Author

Meraj Uddin Mohammed

## Contact

- GitHub: [@merajuddinmohammed](https://github.com/merajuddinmohammed)
- Project: [Amazon Bin Challenge](https://github.com/merajuddinmohammed/Amazon-bin-challenge)

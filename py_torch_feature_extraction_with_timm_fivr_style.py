"""
PyTorch reimplementation of intermediate CNN feature extraction inspired by:
https://github.com/MKLab-ITI/intermediate-cnn-features

Architectures:
- ResNet / Inception / VGG via timm
- ConvNeXt via timm

IMPORTANT:
This script REQUIRES a Python environment with PyTorch and timm installed.
If torch is not available, the script will fail fast with a clear error
instead of crashing during import.

Expected usage environment:
- Local machine / server / conda env / venv / cluster node
- NOT a restricted sandbox without PyTorch

pip install torch torchvision timm pillow numpy tqdm
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# ------------------ DEPENDENCY CHECK ------------------
try:
    import torch
    import timm
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise RuntimeError(
        "PyTorch or timm is not installed in this environment.\n"
        "This script must be run in a Python environment with torch + timm available.\n"
        "Suggested fix:\n"
        "  - Create a virtualenv or conda env\n"
        "  - pip install torch torchvision timm pillow numpy tqdm\n"
        "Original error: " + str(e)
    )

# ---------------------- CONFIG ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# timm model names
MODELS = {
    "resnet": "resnet50",
    "vgg": "vgg16",
    "inception": "inception_v3",
    "convnext": "convnext_base"
}

# ------------------ PREPROCESSING -------------------
def build_transform(model_name):
    cfg = timm.data.resolve_model_data_config(model_name)
    return timm.data.create_transform(**cfg)

# ---------------- FEATURE EXTRACTOR ----------------
class FeatureExtractor(torch.nn.Module):
    """
    CNN backbone with explicit GLOBAL MAX POOLING
    (matches original FIVR TensorFlow implementation)
    """
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=""  # disable built-in pooling
        )

    def forward(self, x):
        x = self.model.forward_features(x)   # (B, C, H, W)
        x = torch.amax(x, dim=(2, 3))         # GLOBAL MAX POOLING → (B, C)
        return x

# ------------------ DATA LOADING --------------------
def load_images(image_dir, transform):
    images = []
    names = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            images.append(transform(img))
            names.append(fname)
    return images, names

# ------------------- EXTRACTION ---------------------
def extract_features(image_dir, arch, out_path):
    """
    Extract FRAME-LEVEL features.

    Output:
        np.ndarray of shape (T, D)
    """
    if arch not in MODELS:
        raise ValueError(f"Unknown architecture: {arch}")

    model_name = MODELS[arch]
    transform = build_transform(model_name)
    model = FeatureExtractor(model_name).to(DEVICE).eval()

    imgs, _ = load_images(image_dir, transform)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), BATCH_SIZE)):
            batch = torch.stack(imgs[i:i+BATCH_SIZE]).to(DEVICE)
            f = model(batch)
            feats.append(f.cpu().numpy())

    feats = np.vstack(feats)
    np.save(out_path, feats)
    print(f"Saved {arch} frame-level features → {out_path} | shape={feats.shape}")
    return feats

# ---------------- TEMPORAL AGGREGATION ----------------
def temporal_aggregate(frame_features, method="mean"):
    """
    TEMPORAL aggregation across frames.

    frame_features: np.ndarray (T, D)
    method:
        - mean    : temporal average pooling (DEFAULT)
        - max     : temporal max pooling
        - l2mean  : L2-normalized temporal mean

    returns:
        np.ndarray (D,)
    """
    if frame_features.ndim != 2:
        raise ValueError("frame_features must be 2D (T, D)")

    if method == "mean":
        return frame_features.mean(axis=0)

    elif method == "max":
        return frame_features.max(axis=0)

    elif method == "l2mean":
        norms = np.linalg.norm(frame_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = frame_features / norms
        v = x.mean(axis=0)
        return v / np.linalg.norm(v)

    else:
        raise ValueError(f"Unknown temporal aggregation method: {method}")

# ---------------- VIDEO-LEVEL PIPELINE ----------------
def extract_video_features(image_dir, arch, out_path, agg="mean"):
    """
    Full pipeline:
    frames → CNN (global MAX pooling) → temporal aggregation
    """
    frame_feats = extract_features(image_dir, arch, out_path + "_frames.npy")
    video_feat = temporal_aggregate(frame_feats, method=agg)
    np.save(out_path, video_feat)
    print(f"Saved {arch} video-level features → {out_path} | shape={video_feat.shape}")

# ---------------------- TESTS ------------------------
def _test_temporal_aggregation():
    """Lightweight unit tests (no torch required)."""
    x = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0]
    ])

    mean_expected = np.array([2.0, 2.0, 2.0])
    max_expected = np.array([3.0, 2.0, 3.0])

    assert np.allclose(temporal_aggregate(x, "mean"), mean_expected)
    assert np.allclose(temporal_aggregate(x, "max"), max_expected)

    v = temporal_aggregate(x, "l2mean")
    assert np.isclose(np.linalg.norm(v), 1.0)

    print("Temporal aggregation tests passed.")

# ---------------------- MAIN ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="Directory of frames")
    parser.add_argument("--arch", choices=MODELS.keys())
    parser.add_argument("--out", help="Output .npy path")
    parser.add_argument("--agg", default="mean", help="Temporal aggregation method")
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()

    if args.run_tests:
        _test_temporal_aggregation()
        sys.exit(0)

    if not args.image_dir or not args.arch or not args.out:
        parser.error("--image_dir, --arch, and --out are required unless --run_tests is used")

    extract_video_features(args.image_dir, args.arch, args.out, args.agg)
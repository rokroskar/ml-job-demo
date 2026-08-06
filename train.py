#!/usr/bin/env python3
"""Train a small PyTorch classifier and write artifacts to an output folder.

The RenkuLab job is configured to read MNIST from a mounted Zenodo data
connector, not by downloading data in the training script. The expected mounted
files are the standard IDX files from Zenodo record 10.5281/zenodo.10058130.

The script also keeps a small CSV mode for trying other connector-mounted data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import random
import struct
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DigitsNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", default=".", help="Directory where artifacts are written.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "uci-optdigits", "sklearn-digits"],
        default="mnist",
        help="Dataset to use when --data-path is not provided. Use 'mnist' with a mounted Zenodo data connector.",
    )
    parser.add_argument(
        "--mnist-path",
        default="/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130",
        help="Directory containing the mounted MNIST IDX files from the Zenodo data connector."
    )
    parser.add_argument(
        "--mnist-source",
        choices=["auto", "connector", "zenodo"],
        default="auto",
        help=(
            "Where to load MNIST from. 'connector' requires --mnist-path to contain the mounted files; "
            "'zenodo' streams files directly from Zenodo; 'auto' uses the connector path if present "
            "and otherwise falls back to Zenodo streaming for local runs."
        ),
    )
    parser.add_argument(
        "--data-url",
        default="https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip",
        help="Fallback public UCI optdigits ZIP file used with --dataset uci-optdigits.",
    )
    parser.add_argument("--data-path", default=None, help="Optional CSV file mounted from a data connector.")
    parser.add_argument("--target-column", default=None, help="Target column in --data-path. Required when --data-path is used.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def open_maybe_gzip(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def find_file(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    available = ", ".join(sorted(p.name for p in root.iterdir())) if root.exists() else "<directory does not exist>"
    raise FileNotFoundError(f"None of {names} found in {root}. Available files: {available}")


def read_idx_images_from_bytes(payload: bytes, source: str) -> np.ndarray:
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) if source.endswith(".gz") else io.BytesIO(payload) as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{source} is not an IDX image file, magic={magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_images, rows, cols)
    return data


def read_idx_labels_from_bytes(payload: bytes, source: str) -> np.ndarray:
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) if source.endswith(".gz") else io.BytesIO(payload) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{source} is not an IDX label file, magic={magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if len(labels) != n_labels:
        raise ValueError(f"Expected {n_labels} labels in {source}, found {len(labels)}")
    return labels.astype(np.int64)


def read_idx_images(path: Path) -> np.ndarray:
    return read_idx_images_from_bytes(path.read_bytes(), str(path))


def read_idx_labels(path: Path) -> np.ndarray:
    return read_idx_labels_from_bytes(path.read_bytes(), str(path))


def mnist_files_available(root: Path) -> bool:
    required = [
        ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"),
        ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"),
        ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"),
        ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"),
    ]
    return root.exists() and all(any((root / name).exists() for name in names) for names in required)


def assemble_mnist(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    images = np.concatenate([train_images, test_images])
    y = np.concatenate([train_labels, test_labels])
    X = images.reshape(len(images), -1).astype(np.float32)
    return X, y, [str(i) for i in range(10)], images


def load_mnist_from_connector(mnist_path: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    root = Path(mnist_path).expanduser()
    train_images = read_idx_images(find_file(root, "train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"))
    train_labels = read_idx_labels(find_file(root, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"))
    test_images = read_idx_images(find_file(root, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"))
    test_labels = read_idx_labels(find_file(root, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"))
    return assemble_mnist(train_images, train_labels, test_images, test_labels)


def download_zenodo_file(name: str) -> bytes:
    url = f"https://zenodo.org/records/10058130/files/{name}?download=1"
    print(f"Streaming {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def load_mnist_from_zenodo() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    train_images = read_idx_images_from_bytes(download_zenodo_file("train-images-idx3-ubyte.gz"), "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels_from_bytes(download_zenodo_file("train-labels-idx1-ubyte.gz"), "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images_from_bytes(download_zenodo_file("t10k-images-idx3-ubyte.gz"), "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels_from_bytes(download_zenodo_file("t10k-labels-idx1-ubyte.gz"), "t10k-labels-idx1-ubyte.gz")
    return assemble_mnist(train_images, train_labels, test_images, test_labels)


def load_connector_csv(data_path: str, target_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(data_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or target_column not in rows[0]:
        raise ValueError(f"Target column '{target_column}' not found in {data_path}")

    feature_names = [name for name in rows[0] if name != target_column]
    numeric_features: list[str] = []
    for name in feature_names:
        try:
            for row in rows:
                if row[name] != "":
                    float(row[name])
            numeric_features.append(name)
        except ValueError:
            pass
    if not numeric_features:
        raise ValueError("No numeric feature columns found in the CSV file")

    X = np.array([[float(row[name] or 0.0) for name in numeric_features] for row in rows], dtype=np.float32)
    labels = [row[target_column] for row in rows]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int64)
    return X, y, [str(c) for c in encoder.classes_]


def load_uci_optdigits(url: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()

    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            text = "\n".join(archive.read(name).decode("utf-8").strip() for name in ["optdigits.tra", "optdigits.tes"])
    else:
        text = payload.decode("utf-8")

    data = np.loadtxt(io.StringIO(text), delimiter=",", dtype=np.float32)
    X = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    return X, y, [str(i) for i in sorted(np.unique(y))]


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str], str, Optional[np.ndarray]]:
    if args.data_path:
        if not args.target_column:
            raise ValueError("--target-column is required when --data-path is provided")
        X, y, class_names = load_connector_csv(args.data_path, args.target_column)
        return X, y, class_names, str(args.data_path), None

    if args.dataset == "mnist":
        root = Path(args.mnist_path).expanduser()
        if args.mnist_source in ("auto", "connector") and mnist_files_available(root):
            X, y, class_names, images = load_mnist_from_connector(args.mnist_path)
            return X, y, class_names, f"Zenodo MNIST data connector at {args.mnist_path}", images
        if args.mnist_source == "connector":
            # Re-run the connector loader to produce a detailed missing-file error.
            X, y, class_names, images = load_mnist_from_connector(args.mnist_path)
            return X, y, class_names, f"Zenodo MNIST data connector at {args.mnist_path}", images
        print(f"MNIST files not found at {args.mnist_path}; streaming from Zenodo for this run.", flush=True)
        X, y, class_names, images = load_mnist_from_zenodo()
        return X, y, class_names, "Zenodo MNIST record 10.5281/zenodo.10058130 streamed over HTTPS", images

    if args.dataset == "uci-optdigits":
        X, y, class_names = load_uci_optdigits(args.data_url)
        return X, y, class_names, args.data_url, None

    digits = load_digits()
    return (
        digits.data.astype(np.float32),
        digits.target.astype(np.int64),
        [str(c) for c in digits.target_names],
        "sklearn.datasets.load_digits",
        digits.images,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets_all = [], []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets_all.append(targets.numpy())
    return np.concatenate(targets_all), np.concatenate(preds)


def draw_training_curve(history: list[dict], path: Path) -> None:
    width, height, pad = 900, 500, 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((pad, pad, width - pad, height - pad), outline="black")
    draw.text((pad, 20), "Training loss (blue) and test accuracy (green)", fill="black")

    def points(key: str) -> list[tuple[int, int]]:
        values = [r[key] for r in history]
        lo, hi = min(values), max(values)
        if hi == lo:
            hi = lo + 1.0
        return [
            (
                pad + int(i * (width - 2 * pad) / max(1, len(history) - 1)),
                height - pad - int((row[key] - lo) * (height - 2 * pad) / (hi - lo)),
            )
            for i, row in enumerate(history)
        ]

    draw.line(points("train_loss"), fill="blue", width=3)
    draw.line(points("test_accuracy"), fill="green", width=3)
    img.save(path)


def draw_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray, class_names: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, predictions)
    cell, label = 46, 80
    size = label + cell * len(class_names) + 20
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    max_value = max(1, int(cm.max()))
    draw.text((label, 20), "Confusion matrix", fill="black")
    for i, actual in enumerate(class_names):
        draw.text((20, label + i * cell + 15), actual, fill="black")
        draw.text((label + i * cell + 15, label - 25), actual, fill="black")
        for j in range(len(class_names)):
            value = int(cm[i, j])
            shade = 255 - int(220 * value / max_value)
            x0, y0 = label + j * cell, label + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(shade, shade, 255), outline="gray")
            draw.text((x0 + 12, y0 + 15), str(value), fill="black")
    img.save(path)


def draw_sample_predictions(images: Optional[np.ndarray], y_true: np.ndarray, predictions: np.ndarray, test_indices: np.ndarray, path: Path) -> None:
    if images is None:
        return
    scale, tile, margin = 6, 190, 20
    img = Image.new("RGB", (4 * tile, 4 * tile), "white")
    draw = ImageDraw.Draw(img)
    for n, (idx, actual, predicted) in enumerate(zip(test_indices[:16], y_true[:16], predictions[:16])):
        x, y = (n % 4) * tile, (n // 4) * tile
        arr = images[idx]
        digit = (255 - (arr / max(1, arr.max()) * 255)).astype(np.uint8)
        digit_img = Image.fromarray(digit, mode="L").resize((digit.shape[1] * scale, digit.shape[0] * scale), Image.Resampling.NEAREST).convert("RGB")
        img.paste(digit_img, (x, y))
        color = "green" if actual == predicted else "red"
        draw.text((x + margin, y + digit_img.height + 3), f"true={actual} pred={predicted}", fill=color)
    img.save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    set_seed(args.random_state)
    output_path = Path(args.output_path).expanduser().resolve() / os.getenv("RENKU_SUBMISSION_ID", "mnist-training") 
    output_path.mkdir(parents=True, exist_ok=True)

    X, y, class_names, source, images = load_data(args)
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, indices, test_size=0.25, random_state=args.random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long()), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitsNet(input_dim=X.shape[1], output_dim=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(features)
        y_eval, pred_eval = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": running_loss / len(X_train), "test_accuracy": accuracy_score(y_eval, pred_eval)}
        history.append(row)
        print(f"epoch={epoch:02d} loss={row['train_loss']:.4f} test_accuracy={row['test_accuracy']:.4f}")

    y_true, predictions = evaluate(model, test_loader, device)
    accuracy = accuracy_score(y_true, predictions)
    elapsed = time.perf_counter() - start_time
    report = classification_report(y_true, predictions, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "data_source": source,
        "device": str(device),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "runtime_seconds": elapsed,
        "epochs": args.epochs,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_dim": int(X.shape[1]),
        "classes": class_names,
        "classification_report": report,
    }

    write_csv(output_path / "training_history.csv", history)
    write_csv(output_path / "predictions.csv", [{"actual": int(a), "predicted": int(p)} for a, p in zip(y_true, predictions)])
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"model_state_dict": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "input_dim": X.shape[1], "classes": class_names}, output_path / "model.pt")

    draw_training_curve(history, output_path / "training_curve.png")
    draw_confusion_matrix(y_true, predictions, class_names, output_path / "confusion_matrix.png")
    draw_sample_predictions(images, y_true, predictions, idx_test, output_path / "sample_predictions.png")

    (output_path / "report.md").write_text(f"""# PyTorch MNIST training job

Data source: `{source}`

- Test accuracy: `{accuracy:.4f}`
- Macro F1 score: `{metrics['macro_f1']:.4f}`
- Weighted F1 score: `{metrics['weighted_f1']:.4f}`
- Runtime: `{elapsed:.1f}` seconds (`{elapsed / 60:.2f}` minutes)
- Epochs: `{args.epochs}`
- Device: `{device}`
- Training rows: `{len(X_train)}`
- Test rows: `{len(X_test)}`

Artifacts: `model.pt`, `metrics.json`, `training_history.csv`, `training_curve.png`, `confusion_matrix.png`, `sample_predictions.png`, `predictions.csv`.
""")
    print("\n=== Job summary ===")
    print(f"Artifacts written to: {output_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro F1 score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 score: {metrics['weighted_f1']:.4f}")
    print(f"Total runtime: {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes)")

    time.sleep(30)
    
if __name__ == "__main__":
    main()

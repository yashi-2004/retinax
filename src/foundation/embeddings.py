"""Unified embedding extraction, caching, and 2D projection diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader


def extract_embeddings_from_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Extract L2-raw embeddings, labels, and image IDs from any RetinaX encoder/model."""
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)

    embs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    ids_list: list[str] = []

    with torch.inference_mode():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, ids = batch
                ids_list.extend([str(i) for i in ids])
            else:
                x, y = batch
                ids_list.extend([str(i) for i in range(len(y))])
            x = x.to(target_device)
            if hasattr(model, "extract_embeddings"):
                feats = model.extract_embeddings(x)
            else:
                feats = model(x)
            embs_list.append(feats.cpu().numpy())
            labels_list.append(y.numpy())

    embs = np.concatenate(embs_list, axis=0) if embs_list else np.empty((0, 1), dtype=np.float32)
    labels = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.int64)
    return {"embeddings": embs, "labels": labels, "image_ids": ids_list}


def compute_2d_projection(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    method: str = "pca",
    seed: int = 42,
) -> dict[str, Any]:
    """Project high-dimensional visual embeddings to 2D via PCA or t-SNE safely."""
    embs = np.asarray(embeddings, dtype=np.float64)
    if len(embs) < 3:
        coords = np.zeros((len(embs), 2), dtype=np.float64)
        return {"coords_2d": coords, "method": method, "silhouette_score": None}

    kind = method.lower()
    if kind == "pca":
        proj = PCA(n_components=2, random_state=seed)
        coords = proj.fit_transform(embs)
    elif kind == "tsne":
        perplexity = min(30.0, max(2.0, (len(embs) - 1) / 3.0))
        proj = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
        coords = proj.fit_transform(embs)
    else:
        raise ValueError(f"Unsupported projection method '{method}'. Choose 'pca' or 'tsne'.")

    sil = None
    if labels is not None and len(np.unique(labels)) >= 2 and len(embs) > len(np.unique(labels)):
        try:
            sil = round(float(silhouette_score(embs, labels)), 4)
        except Exception:
            sil = None

    return {"coords_2d": coords, "method": kind, "silhouette_score": sil}
